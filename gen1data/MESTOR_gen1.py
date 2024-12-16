# in this version,
# BIN0(0.2s) = BIN(0.2s)/BIN.max()*1
# BIN1(0.02s) = BIN(0.02s)/BIN.max()*1
# BIN2(0.02s*5) = BIN(0.02s)/BIN.max()*1  *  BIN0(0.02s*5)


# 3 in channel


from typing import Callable
from pathlib import Path


from torch.utils.data import Dataset


import torch
import h5py


VOC_CLASSES = (    # always index 0
    'car', 'pedestrian')


        
class Gen1H5(Dataset):
    def __init__(
        self,
        file: Path,
        training: bool = False,
        transform: Callable = None,
        num_events: int = 50000,
        time_window: int = 200000,
        augment=False,
        hyp=None,
        rect=False,
        rank=-1,
        task="train",
        data_dict=None,

    ):
        super().__init__()
        self.augment = augment
        self.hyp = hyp
        self.rect = rect
        self.rank = rank
        self.task = task
        self.data_dict = data_dict

        self.main_process = self.rank in (-1, 0)
        self.task = self.task.capitalize()


        if self.task.lower() == "train":
            file = file / ("training.h5")
        elif self.task.lower() == "val":
            file = file / ("validation.h5")
        elif self.task.lower() == "test":
            file = file / ("testing.h5")

        self.h5 = h5py.File(file, 'r')

        self.classes = ["car", "pedestrian"]

        self._file_names = sorted(self.h5.keys())
        self._num_unique_bboxes = [
            len(self.h5[f"{f}/bbox/t_unique"]) for f in self._file_names
        ]

        
        # self._num_unique_bboxes means the number of gtbox-times in every video
        # which is also the number of the dataset
        #sum(self._num_unique_bboxes) 30605
        #len(self._num_unique_bboxes) 470
        #len(dataset) 30605

        self.height = int(self.h5[f"{self._file_names[0]}/events/height"][()])
        self.width = int(self.h5[f"{self._file_names[0]}/events/width"][()])

        self.transform = transform

        self.num_events = num_events
        self.time_window = time_window
        


    def _adjust_bbox(self, bbox: torch.Tensor, left: torch.Tensor, right: torch.Tensor):
        bbox = bbox.copy()
        bbox[:, 3:5] += bbox[:, 1:3]
        bbox[:, 1:3] = np.clip(bbox[:, 1:3], left, right)
        bbox[:, 3:5] = np.clip(bbox[:, 3:5], left, right)
        bbox[:, 3:5] -= bbox[:, 1:3]
        return bbox

    def convert_idx_to_rel_idx(self, idx):
        counter = 0
        # _num_unique_bboxes[0] means the 0 video's gtbox time's number
        # if input idx(the idx of dataset out) lager than this video's gtframe's number,
        # rel_idx will be the idx - this video's gtframe's number
        # which means it's in the next video
        # so,the rel_idx always means the idx in this video
        # for example
        # for input idx = 200
        # idx_out = ;name =  ;self.h5[name] =
        while idx >= self._num_unique_bboxes[counter]:
            idx -= self._num_unique_bboxes[counter]
            counter += 1
        name = self._file_names[counter]
        
        #print('idx',idx)
        #print('name',name)
        #print('self.h5[name]',self.h5[name])
        
        #idx 14   14=200-66-120 ,so in the 3th video "17-04-04_17-31-14_cut_5_500000_60500000",and idx is 14
        #name 17-04-04_17-31-14_cut_5_500000_60500000
        #self.h5[name] <HDF5 group "/17-04-04_17-31-14_cut_5_500000_60500000" (2 members)>
        # and the 2 members is "bbox"and"events"
        
        return idx, self.h5[name], name

    def _load_bbox(self, handle, idx):
        #  bboxes, event_idx = self._load_bbox(handle["bbox"], idx)
        
        #print('handle["offsets"][96]',handle["offsets"][96])
        # offset  is this gtboxtime's ending box's idx, so idx0:idx1 is this gtboxtime's all box's idxs 
        idx0 = 0 if idx == 0 else handle["offsets"][idx - 1]
        idx1 = handle["offsets"][idx]
        bbox = np.stack(
            [
    
                (handle["x"][idx0:idx1].astype("float32")),
                (handle["y"][idx0:idx1].astype("float32")),
                handle["w"][idx0:idx1].astype("float32"),
                handle["h"][idx0:idx1].astype("float32") ,
                handle["class_id"][idx0:idx1].astype("int"),
            ],
            axis=-1,
        )
        event_idx = handle["event_idx"][idx]
        #bbox = self._adjust_bbox(bbox, 0, 1)
        bbox[:, 2:4] += bbox[:, 0:2]

        
        # event_idx is this gtboxtime's event_idx
        return bbox, event_idx

    def _load_events(self, handle, event_idx):
        # event_idx is the idx of gtbox's time's event idx
        idx1 = event_idx
        idx0 = max([0, event_idx - self.num_events])
        # self.num_events is the max event number of read
        # so _load_events will load self.num_events 's event,befor the event_idx

        xyt = np.stack(
            [handle["x"][idx0:idx1], handle["y"][idx0:idx1], handle["t"][idx0:idx1]],
            axis=-1,
        )
        
        polarity = handle["p"][idx0:idx1]

        xyt[:, -1] -= xyt[0, -1]


        return (xyt, polarity)
        
    def _event2frame(self,ev_xyt,time_window,k):

        BIN = np.zeros((240, 304), dtype=np.uint8)
        if k == 0:
            indices_end = -1
        else:
            indices_end = np.where(ev_xyt[:, 2] > (ev_xyt[-1, 2]- (k)*time_window))[0][0]
        
        indices_start = np.where(ev_xyt[:, 2] > (ev_xyt[-1, 2]-(k+1)*time_window))[0][0]
        
        
        
        ev_BIN = ev_xyt[indices_start:indices_end]
        
        for v in range(int(ev_BIN.shape[0])):
            BIN[ev_BIN[v, 1], ev_BIN[v, 0]] += 1


        
        if BIN.max() == 0: 
            pass
        else:
#            BIN = BIN/BIN.max()
            BIN = BIN/BIN.max()*1500  
            BIN[BIN>255] = 255
            BIN = BIN/255
        
        
        #BIN=BIN.astype(np.uint8) 


        return BIN
        
        
    def _event2frame01(self,ev_xyt,time_window):
        # BIN0(0.2s) = BIN(0.2s)/BIN.max()*1
        # BIN1(0.02s) = BIN(0.02s)/BIN.max()*1
        # time_window = 200000
        # time_window = 20000

        
        
        BIN = np.zeros((240, 304), dtype=np.uint8)
        
        indices_end = -1
        indices_start = np.where(ev_xyt[:, 2] > (ev_xyt[-1, 2]-time_window))[0][0]
        
        #print('indices',indices)
        
        ev_BIN = ev_xyt[indices_start:indices_end]
        
        for v in range(int(ev_BIN.shape[0])):
            BIN[ev_BIN[v, 1], ev_BIN[v, 0]] += 1
            
        
        if BIN.max() == 0: 
            pass
        else:
            BIN = BIN/BIN.max()
        
        #BIN=BIN.astype(np.uint8) 
      
        return BIN
        
        
    def _event2frame2(self,ev_xyt,time_window):
        # if k = 4
        # BIN2(0.02s*4) = BIN(0.02s)/BIN.max()*1  with fssnn  BIN(0.02s*4)
        BIN_0 = self._event2frame(ev_xyt,time_window,0)
        BIN_1 = self._event2frame(ev_xyt,time_window,1)
        BIN_2 = self._event2frame(ev_xyt,time_window,2)

        return BIN_0,BIN_1,BIN_2
        
        
    def __len__(self):
        return sum(self._num_unique_bboxes)


    def __getitem__(self, item):
        idx, handle, name = self.convert_idx_to_rel_idx(item)
        # item is 0/1/2/3
        # idx 14   14=200-66-120 ,so in the 3th video "17-04-04_17-31-14_cut_5_500000_60500000",and idx is 14
        # name 17-04-04_17-31-14_cut_5_500000_60500000
        # self.h5[name] <HDF5 group "/17-04-04_17-31-14_cut_5_500000_60500000" (2 members)>
        
        # handle is : self.h5[17-04-04_17-31-14_cut_5_500000_60500000] = <HDF5 group "/17-04-04_17-31-14_cut_5_500000_60500000" (2 members)>
        #  and the 2 members is "bbox"and"events"
        bboxes, event_idx = self._load_bbox(handle["bbox"], idx)
        
        #bbox [[  0 261 130 330 189]
        #[  0  33 144  75 167]]
        # this bbox is [x1,y1,x2,y2]
        #event_idx 17738117
        
        # event_idx is this gtboxtime's event_idx

        
        (ev_xyt, ev_p) = self._load_events(handle["events"], event_idx)

        
# BIN0(0.2s) = BIN(0.2s)/BIN.max()*1
# BIN1(0.02s) = BIN(0.02s)/BIN.max()*1
# BIN2(0.02s*5) = BIN(0.02s)/BIN.max()*1  *  BIN0(0.02s*5)

        BIN0 = self._event2frame(ev_xyt,200000,0)
        BIN1 = self._event2frame(ev_xyt,20000,0)
        BIN_0,BIN_1,BIN_2 = self._event2frame2(ev_xyt,20000)
        

        BIN = np.dstack((BIN0, BIN1, BIN_0,BIN_1,BIN_2))  
        
        bboxes[:,0]=bboxes[:,0]/304
        bboxes[:,1]=bboxes[:,1]/240
        bboxes[:,2]=bboxes[:,2]/304
        bboxes[:,3]=bboxes[:,3]/240     
        
        transform=self.transform
        
        #print('bboxes',bboxes)
    
        BIN, bboxes, labels, scale, offset = transform(BIN, bboxes[:, :4], bboxes[:, 4])   
        
        bboxes = np.hstack((bboxes, np.expand_dims(labels, axis=1)))

        
        return BIN,bboxes

        
    def _getimg(self, item):
        idx, handle, name = self.convert_idx_to_rel_idx(item)
        bboxes, event_idx = self._load_bbox(handle["bbox"], idx)
        (ev_xyt, ev_p) = self._load_events(handle["events"], event_idx)

        BIN0 = self._event2frame(ev_xyt,200000,0)
        BIN1 = self._event2frame(ev_xyt,20000,0)
        BIN_0,BIN_1,BIN_2 = self._event2frame2(ev_xyt,20000)
        
        #BIN = np.stack((BIN0, BIN1 ,BIN2), axis=0) 
        BIN = np.dstack((BIN0, BIN1, BIN_0,BIN_1,BIN_2))  
        
        bboxes[:,0]=bboxes[:,0]/304
        bboxes[:,1]=bboxes[:,1]/240
        bboxes[:,2]=bboxes[:,2]/304
        bboxes[:,3]=bboxes[:,3]/240     
        
        transform=self.transform
        BIN, bboxes, labels, scale, offset = transform(BIN, bboxes[:, :4], bboxes[:, 4])   
        #for trans picdata
        
        
        #return BIN,scale, offset
        
        return BIN
        
    def pull_item_forcoco(self, item):
        idx, handle, name = self.convert_idx_to_rel_idx(item)
        bboxes, event_idx = self._load_bbox(handle["bbox"], idx)
        (ev_xyt, ev_p) = self._load_events(handle["events"], event_idx)
        
  
        
        cocobox = [f"{item}.jpg"]  
  
        
        for bbox in bboxes:  
            bbox_str = [str(int(coord)) for coord in bbox]    
            cocobox.extend(bbox_str)  
        
        #bboxes = np.hstack((bboxes, np.expand_dims(labels, axis=1)))
        
        return cocobox
            
            
if __name__ == "__main__":
    from PIL import Image, ImageDraw 
    from transforms import TrainTransforms, ColorTransforms, ValTransforms
    import numpy as np
    from torch import nn
    import torchvision.transforms as transforms 

    device = torch.device("cuda")
    
    dataset = Gen1H5(
        file=Path("."),
        training=False,
        task="test",
        rank=11,
        num_events= 50000,
        time_window= 200000,
        transform=ValTransforms(640)
    )

    # time_window= 200000us--50hz--0.2s
    # time_window= 20000us--500hz--0.02s
    # time_window= 2000us--5000hz--0.002s


    item = 0
    for item in range(len(dataset)):
        if item%100==0:
            print(item,len(dataset))
#    
        with torch.no_grad():
            K = 3
            alpha = 2
            d0 = H0 = T0 = alpha * 2 ** (-K) * np.array([float(2 ** (K - i)) for i in range(1, K + 1)]).astype(np.float32)
            mem = 0
            spikes = 0
        
            conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, dtype=torch.float).to(device)
        
        
        
            BIN=dataset._getimg(item)
    
            BIN = BIN.unsqueeze(0).to(device)
            
            BIN0 = BIN[:,0:1,:,:]
            BIN1 = BIN[:,1:2,:,:]
            BIN_0 = BIN[:,2:3,:,:]
            BIN_1 = BIN[:,3:4,:,:]
            BIN_2 = BIN[:,4:5,:,:]
            
            init_weights = torch.ones_like(conv1.weight) * 0.15  
     
            conv1.weight.data.copy_(init_weights)  


            ## for K=3
            con_out = conv1(BIN_2)
            c0 = con_out
            mem = mem + c0
            mem = torch.where(mem > T0[0], mem - H0[0], mem)
            spike = torch.zeros_like(mem)
            spike = torch.where(mem > T0[0], 1, spike)
            spikes = spikes + spike
            
            con_out = conv1(BIN_1)
            c0 = con_out
            mem = mem + c0
            mem = torch.where(mem > T0[1], mem - H0[1], mem)
            spike = torch.zeros_like(mem)
            spike = torch.where(mem > T0[1], 1, spike)
            spikes = spikes + spike
            
            con_out = conv1(BIN_0)
            c0 = con_out
            mem = mem + c0
            mem = torch.where(mem > T0[2], mem - H0[2], mem)
            spike = torch.zeros_like(mem)
            spike = torch.where(mem > T0[2], 1, spike)
            spikes = spikes + spike
                
            BIN2 = torch.squeeze(spikes.clone())
            BIN2[BIN2 < K-1] = 0
            # only the spikes number>=K-1(3),can be save
            BIN2[BIN2>0] = 1       
            BIN2 = BIN2.unsqueeze(0)  
            BIN2 = BIN2.unsqueeze(0)    
            
            BIN_IN = torch.cat((BIN0, BIN1, BIN2), dim=1)  
            BIN_IN = BIN_IN.squeeze(0).permute(1, 2, 0)
            
            BIN_IN = BIN_IN.cpu().numpy()*255
            BIN_IN=BIN_IN.astype(np.uint8)
            

            
            to_pil = transforms.ToPILImage()  
            rgb_image = to_pil(BIN_IN)  

            rgb_image.save(f'./a2_testing_3ch/{item}.jpg')


            BIN = None
            conv1 = None
            spikes = None
            mem = None
        

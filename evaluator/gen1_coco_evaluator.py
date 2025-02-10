import json
import tempfile
import torch
#from data.coco import *
from data.gen1 import VOCDetection, VOC_CLASSES
from data.transforms import TrainTransforms, ColorTransforms, ValTransforms
import time
import numpy as np
import os

try:
    from pycocotools.cocoeval import COCOeval
    from pycocotools.coco import COCO
except:
    print("It seems that the COCOAPI is not installed.")

#device = torch.device("cuda")

class COCOAPIEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """

    def __init__(self, data_dir, img_size, device, testset=False, transform=None):
        """
        Args:
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.testset = testset
#        if self.testset:
#            image_set = 'test2017'
#        else:
#            image_set = 'val2017'

        # self.dataset = COCODataset(
        #     data_dir=data_dir,
        #     image_set=image_set,
        #     img_size=img_size,
        #     transform=None)

        self.dataset = VOCDetection(data_dir=data_dir,
                                    image_sets='testing',
                                    img_size = img_size,
                                    transform=ValTransforms(img_size))
                                    #testing
                                    #validation

        self.img_size = img_size
        self.transform = transform
        self.device = device

        self.map = 0.
        self.ap50_95 = 0.
        self.ap50 = 0.

    def evaluate(self, model):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        model.eval()
        
        #model.to(device)
        
        ids = []
        data_dict = []
        num_images = len(self.dataset)
        print('total number of images: %d' % (num_images))
        
        ifsc=0
        #num_images = 5
        
        
        
        annotations = []
        results = []
        images = []
        ann_id = 0

        # start testing
        for index in range(num_images):  # all the data in val2017
        #for index in range(0,2):
        
            id_ = index
            
#            if index % 500 == 0:
#                print('[Eval: %d / %d]' % (index, num_images))

            images.append({
            "date_captured": "2019",
            "file_name": "n.a",
            "id": id_,
            "license": 1,
            "url": "",
            "height": 640,
            "width": 640
            })



            # load an image
            img, target, height, width, scale, offset , cocogtbox= self.dataset.pull_item_forcoco(index)
            # cocogtbox:[(00555_person_badminton_outdoor7_0.jpg,187,136,226,247,6)]

            
            
            num_obj = (len(cocogtbox) - 1) // 5
            for i in range(num_obj):
                x1 = int(cocogtbox[1+5*i])
                y1 = int(cocogtbox[2+5*i])
                x2 = int(cocogtbox[3+5*i])
                y2 = int(cocogtbox[4+5*i])
                label = int(cocogtbox[5+5*i])+1

                bbox = [x1, y1, x2 - x1, y2 - y1]
                area = (x2 - x1) * (y2 - y1)
                
                #print('coco_gtbox',bbox,'coco_gtlabel',label)
                
                annotation = {
                "area": float(area),
                "iscrowd": False,
                "image_id": id_,
                "bbox": bbox,
                "category_id": label,
                "id": ann_id
                }
                
                
                annotations.append(annotation)
                ann_id += 1

            
            img, _ = self.dataset.pull_image(index)
             
            
            h, w, _ = img.shape
            size = np.array([[w, h, w, h]])
            
            #print('size',size)

            # preprocess
            #print('img[img>0]',img[img>0])
            
            x, _, _, scale, offset = self.transform(img)
            x = x.unsqueeze(0).to(self.device)
            id_ = int(id_)
            ids.append(id_)
            
            # inference
            with torch.no_grad():

                
                t0 = time.time()

                outputs = model(x)

                detect_time = time.time() - t0
                bboxes, scores, cls_inds,sc = outputs
                
                if ifsc ==0:
                    SC = sc
                    ifsc =1
                else:
                    SC = SC + sc
                
                
                
                
                
                # map the boxes to original image
                bboxes -= offset
                bboxes /= scale
                bboxes *= size

            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                #label = self.dataset.class_ids[int(cls_inds[i])]
                label = int(cls_inds[i])+1

                bbox = [x1, y1, x2 - x1, y2 - y1]
                
                
                score = float(scores[i])  # object score * class score
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                     "score": score}  # COCO json format

                     
                
                
                
                data_dict.append(A)
                
            if index % 100 == 0:
                print('im_detect: {:d}/{:d} {:.3f}s'.format(index, num_images, detect_time))


            img = None
            target = None
            height = None
            width = None
            scale = None
            offset = None
            cocogtbox = None
            annotation = None
            x = None
            outputs = None
            bboxes = None
            scores = None
            cls_inds = None
            sc = None

                
        dataset = {
        "info": {},
        "licenses": [],
        "type": 'instances',
        "images": images,
        "annotations": annotations,
        "categories": [ {"id": 1, "name": "1", "supercategory": "none"}, 
                       {"id": 2, "name": "2", "supercategory": "none"}  ]
        }        
                
                

        annType = ['segm', 'bbox', 'keypoints']
        
        SC =SC / num_images
        print('fr',SC)
        


        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            print('evaluating ......')
            cocoGt = COCO()
            cocoGt.dataset = dataset
            cocoGt.createIndex()
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.

            
            # _, tmp = tempfile.mkstemp()
            # json.dump(data_dict, open(tmp, 'w'))

            with open('data.json', 'w') as f:
                json.dump(data_dict, f)

            cocoDt = cocoGt.loadRes('data.json')
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            ap50_95, ap50 = cocoEval.stats[0], cocoEval.stats[1]
            print('ap50_95 : ', ap50_95)
            print('ap50 : ', ap50)
            self.map = ap50_95
            self.ap50_95 = ap50_95
            self.ap50 = ap50
            
            #os.remove(tmp)
            os.remove('data.json')

            return ap50, ap50_95
        else:
            return 0, 0


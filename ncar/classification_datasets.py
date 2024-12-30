import os
import random
import tqdm

import torch
from torch.utils.data import Dataset


import argparse

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from prophesee_utils.io.psee_loader import PSEELoader

from PIL import Image, ImageDraw
import torch
from torch import nn
from scipy.ndimage import zoom



from torchvision import transforms



# modified from https://github.com/loiccordone/object-detection-with-spiking-neural-networks/blob/main/datasets/classification_datasets.py
device = torch.device("cuda")

transform = transforms.Compose([
#    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(p=0.5), 
#    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
#    transforms.Resize((64, 64)),  
    transforms.ToTensor(),  
])


class ClassificationDataset(Dataset):
    def __init__(self, args, mode):
        self.mode = mode

        self.sample_size = args.sample_size  # duration of a sample in µs

        save_file_name = f"{args.dataset}_{mode}_{self.sample_size // 1000}.pt"
        save_file = os.path.join(args.path, save_file_name)

        if os.path.isfile(save_file):
            self.samples = torch.load(save_file)
            print("File loaded.")
        else:
            data_dir = os.path.join(args.path, mode)
            self.samples = self.build_dataset(data_dir, save_file)
            torch.save(self.samples, save_file)
            print(f"Done! File saved as {save_file}.")

    def __getitem__(self, index):

        BIN_in, target = self.samples[index]


        if(self.mode=='train'):
            BIN_in = BIN_in *255
            BIN_in = BIN_in.numpy().astype(np.uint8).transpose(1, 2, 0)
            BIN_in = Image.fromarray(BIN_in)
            BIN_in = transform(BIN_in)

        return BIN_in, target

    def __len__(self):
        return len(self.samples)

    def build_dataset(self, data_dir, save_file):
        raise NotImplementedError("The method build_dataset has not been implemented.")


class NCARSClassificationDataset(ClassificationDataset):
    def __init__(self, args, mode="test"):
        super().__init__(args, mode)

    def build_dataset(self, data_dir, save_file):
        classes_dir = [os.path.join(data_dir, class_name) for class_name in os.listdir(data_dir)]
        samples = []
        for class_id, class_dir in enumerate(classes_dir):
            self.files = [os.path.join(class_dir, time_seq_name) for time_seq_name in os.listdir(class_dir)]
            target = class_id

            print(f'Building the class number {class_id + 1}')
            pbar = tqdm.tqdm(total=len(self.files), unit='File', unit_scale=True)

            for i,file_name in enumerate(self.files):

                #print(f"Processing {file_name}...")
                video = PSEELoader(file_name)
                events = video.load_delta_t(self.sample_size)  # Load data

                if events.size == 0:
                    print("Empty sample.")
                    continue

                events['t'] -= events['t'][0]
                

                
                event_BIN0= events
                if event_BIN0.size == 0:
                    print("Empty sample.")
                    continue
                event_BIN0 = structured_to_unstructured(event_BIN0[['y', 'x']], dtype=int)
                BIN0 = np.zeros((event_BIN0[:,0].max(), event_BIN0[:,1].max()), dtype=np.float32)
                for y, x in event_BIN0:
                    BIN0[y-1, x-1] += 1
                BIN0 = zoom(BIN0, zoom=(64 / BIN0.shape[0], 64 / BIN0.shape[1]), order=1)
                BIN0 = BIN0/BIN0.max()*1500  
                BIN0[BIN0>255] = 255
                BIN0 = BIN0/255              
                BIN0 = torch.from_numpy(BIN0)
                

                
                # 100000us=0.1s
                
                event_BIN1= events[events['t']>70000]
                if event_BIN1.size == 0:
                    print("Empty sample.")
                    continue
                event_BIN1 = structured_to_unstructured(event_BIN1[['y', 'x']], dtype=int)
                BIN1 = np.zeros((event_BIN1[:,0].max(), event_BIN1[:,1].max()), dtype=np.float32)
                for y, x in event_BIN1:
                    BIN1[y-1, x-1] += 1
                BIN1 = zoom(BIN1, zoom=(64 / BIN1.shape[0], 64 / BIN1.shape[1]), order=1)
                BIN1 = BIN1/BIN1.max()*1500  
                BIN1[BIN1>255] = 255
                BIN1 = BIN1/255              
                BIN1 = torch.from_numpy(BIN1)
                

                # 30000us=0.03s

                event_BIN2_0= events[(events['t']<30000)]
                if event_BIN2_0.size == 0:
                    print("Empty sample.")
                    continue
                event_BIN2_0 = structured_to_unstructured(event_BIN2_0[['y', 'x']], dtype=int)
                BIN2_0 = np.zeros((event_BIN2_0[:,0].max(), event_BIN2_0[:,1].max()), dtype=np.float32)
                for y, x in event_BIN2_0:
                    BIN2_0[y-1, x-1] += 1
                BIN2_0 = zoom(BIN2_0, zoom=(64 / BIN2_0.shape[0], 64 / BIN2_0.shape[1]), order=1)
                BIN2_0 = BIN2_0/BIN2_0.max()*1500  
                BIN2_0[BIN2_0>255] = 255
                BIN2_0 = BIN2_0/255              
                BIN2_0 = torch.from_numpy(BIN2_0)
                
                event_BIN2_1= events[(events['t']>30000)*(events['t']<60000)]
                if event_BIN2_1.size == 0:
                    print("Empty sample.")
                    continue
                event_BIN2_1 = structured_to_unstructured(event_BIN2_1[['y', 'x']], dtype=int)
                BIN2_1 = np.zeros((event_BIN2_1[:,0].max(), event_BIN2_1[:,1].max()), dtype=np.float32)
                for y, x in event_BIN2_1:
                    BIN2_1[y-1, x-1] += 1
                BIN2_1 = zoom(BIN2_1, zoom=(64 / BIN2_1.shape[0], 64 / BIN2_1.shape[1]), order=1)
                BIN2_1 = BIN2_1/BIN2_1.max()*1500  
                BIN2_1[BIN2_1>255] = 255
                BIN2_1 = BIN2_1/255              
                BIN2_1 = torch.from_numpy(BIN2_1)
                
                event_BIN2_2= events[events['t']>60000]
                if event_BIN2_2.size == 0:
                    print("Empty sample.")
                    continue
                event_BIN2_2 = structured_to_unstructured(event_BIN2_2[['y', 'x']], dtype=int)
                BIN2_2 = np.zeros((event_BIN2_2[:,0].max(), event_BIN2_2[:,1].max()), dtype=np.float32)
                for y, x in event_BIN2_2:
                    BIN2_2[y-1, x-1] += 1
                BIN2_2 = zoom(BIN2_2, zoom=(64 / BIN2_2.shape[0], 64 / BIN2_2.shape[1]), order=1)
                BIN2_2 = BIN2_2/BIN2_2.max()*1500  
                BIN2_2[BIN2_2>255] = 255
                BIN2_2 = BIN2_2/255              
                BIN2_2 = torch.from_numpy(BIN2_2)
                # 30000us*3=0.03s*3
                
                with torch.no_grad():
                    K = 3
                    alpha = 2
                    d0 = H0 = T0 = alpha * 2 ** (-K) * np.array([float(2 ** (K - i)) for i in range(1, K + 1)]).astype(np.float32)
                    mem = 0
                    spikes = 0
                
                    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, dtype=torch.float).to(device)
            
                    BIN2_0 = BIN2_0.unsqueeze(0).unsqueeze(0).to(device)
                    BIN2_1 = BIN2_1.unsqueeze(0).unsqueeze(0).to(device)
                    BIN2_2 = BIN2_2.unsqueeze(0).unsqueeze(0).to(device)
                    
                    
                    init_weights = torch.ones_like(conv1.weight) * 0.15  
             
                    conv1.weight.data.copy_(init_weights)  
                    
                    con_out = conv1(BIN2_0)
                    c0 = con_out
                    mem = mem + c0
                    mem = torch.where(mem > T0[0], mem - H0[0], mem)
                    spike = torch.zeros_like(mem)
                    spike = torch.where(mem > T0[0], 1, spike)
                    spikes = spikes + spike
                    
                    con_out = conv1(BIN2_1)
                    c0 = con_out
                    mem = mem + c0
                    mem = torch.where(mem > T0[1], mem - H0[1], mem)
                    spike = torch.zeros_like(mem)
                    spike = torch.where(mem > T0[1], 1, spike)
                    spikes = spikes + spike
                    
                    con_out = conv1(BIN2_2)
                    c0 = con_out
                    mem = mem + c0
                    mem = torch.where(mem > T0[2], mem - H0[2], mem)
                    spike = torch.zeros_like(mem)
                    spike = torch.where(mem > T0[2], 1, spike)
                    spikes = spikes + spike
                        
                    BIN2 = torch.squeeze(spikes.clone()).cpu()
                    BIN2[BIN2 < K-1] = 0
                    # only the spikes number>=K-1(3),can be save
                    BIN2[BIN2>0] = 1      


                BIN_in = torch.stack((BIN0, BIN1, BIN2), dim=0)
                samples.append([BIN_in, target])

                pbar.update(1)

            pbar.close()

        return samples
        
def main():    
    parser = argparse.ArgumentParser(description='Classify event dataset')

    parser.add_argument('-dataset', default='ncars', type=str, help='dataset used NCAR')
    parser.add_argument('-path', default='./Prophesee_Dataset_n_cars/', type=str, help='dataset used. NCAR')
    parser.add_argument('-sample_size', default=100000, type=int, help='duration of a sample in µs')

    args = parser.parse_args()
    
    dataset = NCARSClassificationDataset  
    test_dataset = dataset(args, mode="test")  ##test train




if __name__ == '__main__':
    main()
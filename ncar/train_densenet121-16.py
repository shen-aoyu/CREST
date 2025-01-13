#coding=utf-8


# 1./0.
import os
import torch
import torch.nn as nn
import numpy as np  
import time
#from torchstat import stat

from PIL import Image, ImageDraw
import argparse
from classification_datasets import NCARSClassificationDataset

import torch.optim as optim

from thop import profile

device = torch.device("cuda")
lens = 0.5  # hyper-parameters of approximate function




K = [0, 5, 5, 5, 5,    5, 5, 5, 5, 5,    5, 5, 5, 5, 5,    5, 5, 5, 5, 5,    5, 5, 5, 5, 5,   5, 5, 5, 5, 5,
      5, 5, 5, 5, 5,    5, 5, 5, 5, 5,    5, 5, 5, 5, 5,    5, 5, 5, 5, 5,   5, 5, 5, 5, 5,   5, 5, 5, 5, 5,
      5, 5, 5, 5, 5]
      
alpha = [0, 1, 3, 3, 3,    3, 3, 3, 3, 3,    3, 3, 3, 3, 3,    3, 3, 3, 3, 3,    3, 3, 3, 3, 3,   3, 3, 3, 3, 3,
      3, 3, 3, 3, 3,    3, 3, 3, 3, 3,    3, 3, 3, 3, 3,    3, 3, 3, 3, 3,   3, 3, 3, 3, 3,   3, 3, 3, 3, 3,
      3, 3, 5, 5, 5]





#D0 = H0 = T0 = alpha[0] * 2 ** (-K[0]) * np.array([float(2 ** (K[0] - i)) for i in range(1, K[0] + 1)]).astype(np.float32)
D1 = H1 = T1 = alpha[1] * 2 ** (-K[1]) * np.array([float(2 ** (K[1] - i)) for i in range(1, K[1] + 1)]).astype(np.float32)
D2 = H2 = T2 = alpha[2] * 2 ** (-K[2]) * np.array([float(2 ** (K[2] - i)) for i in range(1, K[2] + 1)]).astype(np.float32)
D3 = H3 = T3 = alpha[3] * 2 ** (-K[3]) * np.array([float(2 ** (K[3] - i)) for i in range(1, K[3] + 1)]).astype(np.float32)
D4 = H4 = T4 = alpha[4] * 2 ** (-K[4]) * np.array([float(2 ** (K[4] - i)) for i in range(1, K[4] + 1)]).astype(np.float32)
D5 = H5 = T5 = alpha[5] * 2 ** (-K[5]) * np.array([float(2 ** (K[5] - i)) for i in range(1, K[5] + 1)]).astype(np.float32)
D6 = H6 = T6 = alpha[6] * 2 ** (-K[6]) * np.array([float(2 ** (K[6] - i)) for i in range(1, K[6] + 1)]).astype(np.float32)
D7 = H7 = T7 = alpha[7] * 2 ** (-K[7]) * np.array([float(2 ** (K[7] - i)) for i in range(1, K[7] + 1)]).astype(np.float32)
D8 = H8 = T8 = alpha[8] * 2 ** (-K[8]) * np.array([float(2 ** (K[8] - i)) for i in range(1, K[8] + 1)]).astype(np.float32)
D9 = H9 = T9 = alpha[9] * 2 ** (-K[9]) * np.array([float(2 ** (K[9] - i)) for i in range(1, K[9] + 1)]).astype(np.float32)
D10 = H10 = T10 = alpha[10] * 2 ** (-K[10]) * np.array([float(2 ** (K[10] - i)) for i in range(1, K[10] + 1)]).astype(np.float32)
D11 = H11 = T11 = alpha[11] * 2 ** (-K[11]) * np.array([float(2 ** (K[11] - i)) for i in range(1, K[11] + 1)]).astype(np.float32)
D12 = H12 = T12 = alpha[12] * 2 ** (-K[12]) * np.array([float(2 ** (K[12] - i)) for i in range(1, K[12] + 1)]).astype(np.float32)
D13 = H13 = T13 = alpha[13] * 2 ** (-K[13]) * np.array([float(2 ** (K[13] - i)) for i in range(1, K[13] + 1)]).astype(np.float32)
D14 = H14 = T14 = alpha[14] * 2 ** (-K[14]) * np.array([float(2 ** (K[14] - i)) for i in range(1, K[14] + 1)]).astype(np.float32)
D15 = H15 = T15 = alpha[15] * 2 ** (-K[15]) * np.array([float(2 ** (K[15] - i)) for i in range(1, K[15] + 1)]).astype(np.float32)
D16 = H16 = T16 = alpha[16] * 2 ** (-K[15]) * np.array([float(2 ** (K[16] - i)) for i in range(1, K[16] + 1)]).astype(np.float32)
D17 = H17 = T17 = alpha[17] * 2 ** (-K[17]) * np.array([float(2 ** (K[17] - i)) for i in range(1, K[17] + 1)]).astype(np.float32)
D18 = H18 = T18 = alpha[18] * 2 ** (-K[18]) * np.array([float(2 ** (K[18] - i)) for i in range(1, K[18] + 1)]).astype(np.float32)
D19 = H19 = T19 = alpha[19] * 2 ** (-K[19]) * np.array([float(2 ** (K[19] - i)) for i in range(1, K[19] + 1)]).astype(np.float32)
D20 = H20 = T20 = alpha[20] * 2 ** (-K[20]) * np.array([float(2 ** (K[20] - i)) for i in range(1, K[20] + 1)]).astype(np.float32)
D21 = H21 = T21 = alpha[21] * 2 ** (-K[21]) * np.array([float(2 ** (K[21] - i)) for i in range(1, K[21] + 1)]).astype(np.float32)
D22 = H22 = T22 = alpha[22] * 2 ** (-K[22]) * np.array([float(2 ** (K[22] - i)) for i in range(1, K[22] + 1)]).astype(np.float32)
D23 = H23 = T23 = alpha[23] * 2 ** (-K[23]) * np.array([float(2 ** (K[23] - i)) for i in range(1, K[23] + 1)]).astype(np.float32)
D24 = H24 = T24 = alpha[24] * 2 ** (-K[24]) * np.array([float(2 ** (K[24] - i)) for i in range(1, K[24] + 1)]).astype(np.float32)
D25 = H25 = T25 = alpha[25] * 2 ** (-K[25]) * np.array([float(2 ** (K[25] - i)) for i in range(1, K[25] + 1)]).astype(np.float32)
D26 = H26 = T26 = alpha[26] * 2 ** (-K[25]) * np.array([float(2 ** (K[26] - i)) for i in range(1, K[26] + 1)]).astype(np.float32)
D27 = H27 = T27 = alpha[27] * 2 ** (-K[27]) * np.array([float(2 ** (K[27] - i)) for i in range(1, K[27] + 1)]).astype(np.float32)
D28 = H28 = T28 = alpha[28] * 2 ** (-K[28]) * np.array([float(2 ** (K[28] - i)) for i in range(1, K[28] + 1)]).astype(np.float32)
D29 = H29 = T29 = alpha[29] * 2 ** (-K[29]) * np.array([float(2 ** (K[29] - i)) for i in range(1, K[29] + 1)]).astype(np.float32)
D30 = alpha[30] * 2 ** (-K[30]) * np.array([float(2 ** (K[30] - i)) for i in range(1, K[30] + 1)]).astype(np.float32)
D31 = alpha[31] * 2 ** (-K[31]) * np.array([float(2 ** (K[31] - i)) for i in range(1, K[31] + 1)]).astype(np.float32)
D32 = alpha[32] * 2 ** (-K[32]) * np.array([float(2 ** (K[32] - i)) for i in range(1, K[32] + 1)]).astype(np.float32)
D33 = alpha[33] * 2 ** (-K[33]) * np.array([float(2 ** (K[33] - i)) for i in range(1, K[33] + 1)]).astype(np.float32)
D34 = alpha[34] * 2 ** (-K[34]) * np.array([float(2 ** (K[34] - i)) for i in range(1, K[34] + 1)]).astype(np.float32)
D35 = alpha[35] * 2 ** (-K[35]) * np.array([float(2 ** (K[35] - i)) for i in range(1, K[35] + 1)]).astype(np.float32)
D36 = alpha[36] * 2 ** (-K[35]) * np.array([float(2 ** (K[36] - i)) for i in range(1, K[36] + 1)]).astype(np.float32)
D37 = alpha[37] * 2 ** (-K[37]) * np.array([float(2 ** (K[37] - i)) for i in range(1, K[37] + 1)]).astype(np.float32)
D38 = alpha[38] * 2 ** (-K[38]) * np.array([float(2 ** (K[38] - i)) for i in range(1, K[38] + 1)]).astype(np.float32)
D39 = alpha[39] * 2 ** (-K[39]) * np.array([float(2 ** (K[39] - i)) for i in range(1, K[39] + 1)]).astype(np.float32)
D40 = alpha[40] * 2 ** (-K[40]) * np.array([float(2 ** (K[40] - i)) for i in range(1, K[40] + 1)]).astype(np.float32)
D41 = alpha[41] * 2 ** (-K[41]) * np.array([float(2 ** (K[41] - i)) for i in range(1, K[41] + 1)]).astype(np.float32)
D42 = alpha[42] * 2 ** (-K[42]) * np.array([float(2 ** (K[42] - i)) for i in range(1, K[42] + 1)]).astype(np.float32)
D43 = alpha[43] * 2 ** (-K[43]) * np.array([float(2 ** (K[43] - i)) for i in range(1, K[43] + 1)]).astype(np.float32)
D44 = alpha[44] * 2 ** (-K[44]) * np.array([float(2 ** (K[44] - i)) for i in range(1, K[44] + 1)]).astype(np.float32)
D45 = alpha[45] * 2 ** (-K[45]) * np.array([float(2 ** (K[45] - i)) for i in range(1, K[45] + 1)]).astype(np.float32)
D46 = alpha[46] * 2 ** (-K[45]) * np.array([float(2 ** (K[46] - i)) for i in range(1, K[46] + 1)]).astype(np.float32)
D47 = alpha[47] * 2 ** (-K[47]) * np.array([float(2 ** (K[47] - i)) for i in range(1, K[47] + 1)]).astype(np.float32)
D48 = alpha[48] * 2 ** (-K[48]) * np.array([float(2 ** (K[48] - i)) for i in range(1, K[48] + 1)]).astype(np.float32)
D49 = alpha[49] * 2 ** (-K[49]) * np.array([float(2 ** (K[49] - i)) for i in range(1, K[49] + 1)]).astype(np.float32)
D50 = alpha[50] * 2 ** (-K[50]) * np.array([float(2 ** (K[50] - i)) for i in range(1, K[50] + 1)]).astype(np.float32)
D51 = alpha[51] * 2 ** (-K[51]) * np.array([float(2 ** (K[51] - i)) for i in range(1, K[51] + 1)]).astype(np.float32)
D52 = alpha[52] * 2 ** (-K[52]) * np.array([float(2 ** (K[52] - i)) for i in range(1, K[52] + 1)]).astype(np.float32)
D53 = alpha[53] * 2 ** (-K[53]) * np.array([float(2 ** (K[53] - i)) for i in range(1, K[53] + 1)]).astype(np.float32)
D54 = alpha[54] * 2 ** (-K[54]) * np.array([float(2 ** (K[54] - i)) for i in range(1, K[54] + 1)]).astype(np.float32)
D55 = alpha[55] * 2 ** (-K[55]) * np.array([float(2 ** (K[55] - i)) for i in range(1, K[55] + 1)]).astype(np.float32)
D56 = alpha[56] * 2 ** (-K[56]) * np.array([float(2 ** (K[56] - i)) for i in range(1, K[56] + 1)]).astype(np.float32)
D57 = alpha[57] * 2 ** (-K[57]) * np.array([float(2 ** (K[57] - i)) for i in range(1, K[57] + 1)]).astype(np.float32)
D58 = alpha[58] * 2 ** (-K[58]) * np.array([float(2 ** (K[58] - i)) for i in range(1, K[58] + 1)]).astype(np.float32)
D59 = alpha[59] * 2 ** (-K[59]) * np.array([float(2 ** (K[59] - i)) for i in range(1, K[59] + 1)]).astype(np.float32)
D60 = alpha[60] * 2 ** (-K[60]) * np.array([float(2 ** (K[60] - i)) for i in range(1, K[60] + 1)]).astype(np.float32)
D61 = alpha[61] * 2 ** (-K[61]) * np.array([float(2 ** (K[61] - i)) for i in range(1, K[61] + 1)]).astype(np.float32)
D62 = alpha[62] * 2 ** (-K[62]) * np.array([float(2 ** (K[62] - i)) for i in range(1, K[62] + 1)]).astype(np.float32)
D63 = alpha[63] * 2 ** (-K[63]) * np.array([float(2 ** (K[63] - i)) for i in range(1, K[63] + 1)]).astype(np.float32)
D64 = alpha[64] * 2 ** (-K[64]) * np.array([float(2 ** (K[64] - i)) for i in range(1, K[64] + 1)]).astype(np.float32)


class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)  
        return torch.floor(x)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = 1.0
        return grad_input*temp   
Ops_Q=FakeQuantize.apply

               
# FS Conv block
class FC(nn.Module):
    def __init__(self, c1, c2, K=5,D=[1.5,0.75,0.3725,0.18625,0.093125]):
        super(FC, self).__init__()
        self.fc = nn.Linear(c1, c2, bias=False)
        self.K = K
        self.D = D

        self.Hardtanh = nn.Hardtanh(min_val=0, max_val=self.D[0]*2-self.D[-1])     
        self.c2 = c2
        

    def forward(self, x):
        c = x
        c = self.Hardtanh(c)
        c_spikes = torch.zeros_like(c).detach()
        c_spike = c_spikes
        minmin = self.D[-1]
        num = Ops_Q(c / minmin)
        c = minmin * num
        c_out = self.fc(c)
        return c_out,c_spikes.sum(),c_spike.numel() 
        
# FS Conv block
class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, bias=False,K=5,D=[1.5,0.75,0.3725,0.18625,0.093125]):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False)
        self.K = K
        self.D = D
        self.Hardtanh = nn.Hardtanh(min_val=0, max_val=self.D[0]*2-self.D[-1])
        self.p = p
        self.k = k
        self.s = s
        
        self.c2 = c2
        

    def forward(self, x):
        c = x
        c = self.Hardtanh(c)
        c_spikes = torch.zeros_like(c).detach()
        c_spike = c_spikes
        minmin = self.D[-1]
        num = Ops_Q(c / minmin)
        c = minmin * num
        c_out = self.conv(c)
        return c_out,c_spikes.sum(),c_spike.numel()        
        
class DenseLayer(nn.Module):
    def __init__(self, inplace, growth_rate, bn_size,drop_rate=0):
        super(DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        
        self.bn1=nn.BatchNorm2d(inplace)
        self.relu1=nn.ReLU(inplace=True)
        self.cv1=Conv(c1=inplace, c2=bn_size * growth_rate,s=1,p=0, k=1)
        self.bn2=nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2=nn.ReLU(inplace=True)
        self.cv2=Conv(c1=bn_size * growth_rate, c2=growth_rate,s=1,p=1, k=3)
        
        self.dropout = nn.Dropout(p=self.drop_rate)
 
    def forward(self, x):
        y = self.bn1(x)
        y = self.relu1(y)
        y,c_spikes,c_spike_n = self.cv1(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y,c_spikes1,c_spike_n1 = self.cv2(y)

        if self.drop_rate > 0:
            y = self.dropout(y)
            
        c_spikes =c_spikes + c_spikes1 
        c_spike_n =c_spikes + c_spike_n1   
        
        return torch.cat([x, y], 1),c_spikes,c_spike_n
        
        
        
class DenseBlock(nn.Module):
    def __init__(self, num_layers, inplances, growth_rate, bn_size , drop_rate=0):
        super(DenseBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                DenseLayer(inplances + i * growth_rate, growth_rate, bn_size, drop_rate)
            ) 
 

    def forward(self, x):       
        c_spikes = 0
        c_spike_n = 0
        for layer in self.layers:
            x,c_spikes_t,c_spike_n_t = layer(x)
            c_spikes+=c_spikes_t
            c_spike_n+=c_spike_n_t  
            
        return x,c_spikes,c_spike_n       
        
        
class TransitionLayer(nn.Module):
    def __init__(self, inplace, plance):
        super(TransitionLayer, self).__init__()
        
        self.bn1=nn.BatchNorm2d(inplace)
        self.relu1=nn.ReLU(inplace=True)
        self.cv1=Conv(c1=inplace, c2=plance,s=1,p=0, k=1)
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
   
 
    def forward(self, x):
    
        y = self.bn1(x)
        y = self.relu1(y)
        y,c_spikes,c_spike_n = self.cv1(y)
        y = self.maxpool(y)

        return y,c_spikes,c_spike_n
            
            
            
            
        
class DenseNet(nn.Module):
    def __init__(self,trainable=False, growth_rate=16, blocks=[6, 12, 24, 16],num_classes=2):
        super(DenseNet, self).__init__()
        self.trainable = trainable
        
        bn_size = 4
        drop_rate = 0
        
        self.cv1 = Conv(3, 64, k=3, p=1,s=2, bias=False,K=5,D=[0.5,0.25,0.125,0.0625,0.03125])
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1=nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        num_features = 64

        self.layer1 = DenseBlock(num_layers=blocks[0], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate

        self.transition1 = TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2
 

        self.layer2 = DenseBlock(num_layers=blocks[1], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        self.transition2 = TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2
 

        self.layer3 = DenseBlock(num_layers=blocks[2], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        self.transition3 = TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2
 

        self.layer4 = DenseBlock(num_layers=blocks[3], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.fc1 = FC(num_features, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2 = FC(256, 2)           


    @torch.no_grad()
    def inference_single_image(self, x):

        # backbone
        c3, c4, c5,[c_spikes1,c_spikes2,c_spikes3,c_spikes4,c_spikes5,c_spikes6,c_spikes7,c_spikes8,c_spikes9,c_spikes10,c_spikes11,c_spikes12,c_spikes13,c_spikes14,c_spikes15,c_spikes16,c_spikes17,c_spikes18,c_spikes19,c_spikes20,c_spikes21,c_spikes22,c_spikes23,c_spikes24,c_spikes25,c_spikes26,c_spikes27,c_spikes28,c_spikes29,c_spikes30,c_spikes31] = self.backbone(x)
        
        c5 = c5.view(x.size(0), -1)
        cout,c_spikes32 = self.fc1(c5)
        cout = self.bn1(cout)
        cout,c_spikes33 = self.fc2(cout)

        return cout, 0


    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            gtimg = x.clone().detach()
            
            c = x
            
            cout,_,_ =  self.cv1(c)

            cout =  self.bn1(cout)
            cout =  self.relu1(cout)
            cout =  self.maxpool(cout)
            
            cout,_,_ = self.layer1(cout)
            cout,_,_ = self.transition1(cout)
            
            cout,_,_ = self.layer2(cout)

            cout,_,_ = self.transition2(cout)
            
            
            cout,_,_ = self.layer3(cout)

            cout,_,_ = self.transition3(cout)
            
            cout,_,_ = self.layer4(cout)

            cout =  self.maxpool2(cout)

    
            cout = cout.view(x.size(0), -1)
            cout,_,_=self.fc1(cout)
            cout=self.bn2(cout)
            cout,_,_=self.fc2(cout)

     
            return cout

def FLOPs_and_Params(model, size):

    x = torch.randn(1, 3, size, size).to(device)

    flops, params = profile(model, inputs=(x, ))
    print('MACs : ', flops / 1e9, ' G')
    print('Params : ', params / 1e6, ' M')
    
    
def main():  
        
    model = DenseNet(trainable=True)
    
    

    cnn = model.to(device)

    
    
    FLOPs_and_Params(cnn, 64)

    parser = argparse.ArgumentParser(description='Classify event dataset')
    parser.add_argument('-num_workers', default=4, type=int, help='The number of workers')

    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-sample_size', default=100000, type=int, help='duration of a sample in us')
    parser.add_argument('-epochs', default=100, type=int, help='number of total epochs to run')

    parser.add_argument('-dataset', default='ncars', type=str, help='dataset used NCAR')
    parser.add_argument('-path', default='./Prophesee_Dataset_n_cars/', type=str, help='dataset used. NCAR')
    args = parser.parse_args()
    save_path = './weight/'
    
    dataset = NCARSClassificationDataset  
    test_dataset = dataset(args, mode="test") 
    train_dataset = dataset(args, mode="train") 
    
    batch_size=args.b
    num_workers = args.num_workers
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=True)
    

    
    acc_record = list([])
    loss_train_record = list([])
    epoch_list = []
    num_epochs = args.epochs
    best_epoch=0
    best_acc = 0
    start_epoch = 0
    

    
    
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5, last_epoch=-1)



    
    for epoch in range(num_epochs):
        epoch_list.append(epoch)
        running_loss = 0
        start_time = time.time()
        print('\n\n\n', 'Iters:', epoch)
    
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            
            # cnn.zero_grad()
            optimizer.zero_grad()
            images = images.float().to(device)
            labels = labels.to(device)

    
            outputs = cnn(images)

            loss = criterion(outputs, labels)
            
            running_loss += loss.item()  # 
            loss.backward()  # 
            optimizer.step()  # 
    
            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f,Time elasped: %.5f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, running_loss / 100,
                         time.time() - start_time))
                running_loss = 0

    
        scheduler.step()
        correct = 0
        total = 0

        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs   = cnn(inputs)

                
                _, predicted = outputs.max(1)
                total += float(targets.size(0))
                correct += float(predicted.eq(targets).sum().item())

        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)
        

    
        print('\n', 'Best-epoch/test Accuracy of the cnn on the 10000 test images: %.3f-%d/%.3f' % (best_acc,best_epoch,acc))
    
        print("average spikes number: ", '\n')

        print('epoch_time',time.time() - start_time)
        if acc > best_acc:
            best_acc = acc
            best_epoch =epoch
            print('Saving..')
            checkpoint = {
                "net": cnn.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'scheduler': scheduler.state_dict()
            }
            torch.save(checkpoint, save_path+'_'+str(round(best_acc,3))+'_'+'_'+str(best_epoch)+'.pth')
            print("Saved PyTorch Model State to ", save_path+'_'+str(round(best_acc,3))+'_'+'_'+str(best_epoch)+'.pth')   
        
if __name__ == '__main__':
    main()
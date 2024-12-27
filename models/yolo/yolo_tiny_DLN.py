
# 1./0.
import os
import torch
import torch.nn as nn
import numpy as np  
import time
from utils import box_ops
from PIL import Image, ImageDraw


batch_size = 32
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
class Conv(nn.Module):
    def __init__(self, c1, c2, k, K,D,s=1, p=0, d=1, g=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        
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
        minmin = self.D[-1]
        num = Ops_Q(c / minmin)
        c = minmin * num
        c_out = self.conv(c)
        c_spikes = torch.zeros_like(c)
        c_out  = self.bn(c_out)
        return c_out,c_spikes
        
# FS Conv block
class Conv0(nn.Module):
    def __init__(self, c1, c2, k, K,D,s=1, p=0, d=1, g=1):
        super(Conv0, self).__init__()
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
        minmin = self.D[-1]
        num = Ops_Q(c / minmin)
        c = minmin * num
        c_out = self.conv(c)
        c_spikes = torch.zeros_like(c)
        return c_out,c_spikes
        
        
        
class ResidualBlock(nn.Module):
    """
    basic residual block for CSP-Darknet
    """
    def __init__(self, in_ch,K1,D1,K2,D2):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv(in_ch, in_ch, k=1,K=K1,D=D1)
        self.conv2 = Conv(in_ch, in_ch, k=3, p=1,K=K2,D=D2)

    def forward(self, x):
        h,c_spikes1 = self.conv1(x)
        h,c_spikes2 = self.conv2(h)
        out = x + h

        return out,[c_spikes1,c_spikes2]
        
class CSPStage(nn.Module):
    def __init__(self,K1,D1,K2,D2,K3,D3,K4,D4,K5,D5,c1, n=1):
        super(CSPStage, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k=1,K=K1,D=D1)
        self.cv2 = Conv(c1, c_, k=1,K=K2,D=D2)
        #self.res_blocks = nn.Sequential(*[ResidualBlock(in_ch=c_,K1=K3,D1=D3,K2=K4,D2=D4) for _ in range(n)])
        self.res_blocks = ResidualBlock(in_ch=c_,K1=K3,D1=D3,K2=K4,D2=D4)
        self.cv3 = Conv(2 * c_, c1, k=1,K=K5,D=D5)

    def forward(self, x):
        y1,c_spikes1 = self.cv1(x)
        y2,c_spikes2 = self.cv2(x)
        y2,[c_spikes3,c_spikes4] = self.res_blocks(y2)
        y3,c_spikes5 = self.cv3(torch.cat([y1, y2], dim=1))

        return y3,[c_spikes1,c_spikes2,c_spikes3,c_spikes4,c_spikes5]
        
# CSPDarkNet-Tiny
class CSPDarknetTiny(nn.Module):
    """
    CSPDarknet_Tiny.
    """
    def __init__(self,K1,D1,K2,D2,K3,D3,K4,D4,K5,D5,K6,D6,K7,D7,K8,D8,K9,D9,K10,D10,K11,D11,K12,D12,K13,D13,K14,D14,K15,D15,K16,D16,K17,D17,K18,D18,K19,D19,K20,D20,K21,D21,K22,D22,K23,D23,K24,D24,K25,D25,K26,D26,K27,D27,K28,D28,K29,D29,K30,D30,K31,D31):
    
        super(CSPDarknetTiny, self).__init__()
            
        
        self.l1_1 = Conv(3, 16, k=3, p=1,K=K1,D=D1)      
        self.l1_2 = Conv(16, 32, k=3, p=1, s=2,K=K2,D=D2)
        self.l1_3 = CSPStage(K3,D3,K4,D4,K5,D5,K6,D6,K7,D7,c1=32)                       
        # p1/2

 
        self.l2_1 = Conv(32, 64, k=3, p=1, s=2,K=K8,D=D8)             
        self.l2_2 = CSPStage(K9,D9,K10,D10,K11,D11,K12,D12,K13,D13,c1=64)                      
        # P2/4

            
        self.l3_1 = Conv(64, 128, k=3, p=1, s=2,K=K14,D=D14)             
        self.l3_2 = CSPStage(K15,D15,K16,D16,K17,D17,K18,D18,K19,D19,c1=128)             
        # P3/8
        
                    
        self.l4_1 = Conv(128, 256, k=3, p=1, s=2,K=K20,D=D20)             
        self.l4_2 = CSPStage(K21,D21,K22,D22,K23,D23,K24,D24,K25,D25,c1=256)          
        # P4/16
        
  
        self.l5_1 = Conv(256, 512, k=3, p=1, s=2,K=K26,D=D26)             
        self.l5_2 = CSPStage(K27,D27,K28,D28,K29,D29,K30,D30,K31,D31,c1=512)                    
        # P5/32
        


    def forward(self, x):
        c1,c_spikes1 = self.l1_1(x)
        c1,c_spikes2 = self.l1_2(c1)
        c1,[c_spikes3,c_spikes4,c_spikes5,c_spikes6,c_spikes7]=self.l1_3(c1)
        
        c2,c_spikes8 = self.l2_1(c1)
        c2,[c_spikes9,c_spikes10,c_spikes11,c_spikes12,c_spikes13]=self.l2_2(c2)
        
        c3,c_spikes14 = self.l3_1(c2)
        c3,[c_spikes15,c_spikes16,c_spikes17,c_spikes18,c_spikes19]=self.l3_2(c3)
        
        c4,c_spikes20 = self.l4_1(c3)
        c4,[c_spikes21,c_spikes22,c_spikes23,c_spikes24,c_spikes25]=self.l4_2(c4)
        
        c5,c_spikes26 = self.l5_1(c4)
        c5,[c_spikes27,c_spikes28,c_spikes29,c_spikes30,c_spikes31]=self.l5_2(c5)

        return c3, c4, c5,[c_spikes1,c_spikes2,c_spikes3,c_spikes4,c_spikes5,c_spikes6,c_spikes7,c_spikes8,c_spikes9,c_spikes10,c_spikes11,c_spikes12,c_spikes13,c_spikes14,c_spikes15,c_spikes16,c_spikes17,c_spikes18,c_spikes19,c_spikes20,c_spikes21,c_spikes22,c_spikes23,c_spikes24,c_spikes25,c_spikes26,c_spikes27,c_spikes28,c_spikes29,c_spikes30,c_spikes31]

        


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self, c1, c2, K1,D1,K2,D2,e=0.5, kernel_sizes=[5, 9, 13]):
        super(SPP, self).__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, k=1,K=K1,D=D1)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        
        self.cv2 = Conv(c_*(len(kernel_sizes) + 1), c2, k=1,K=K2,D=D2)

    def forward(self, x):
        x,c_spikes1 = self.cv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x,c_spikes2 = self.cv2(x)

        return x,[c_spikes1,c_spikes2]

class SPPBlockCSP(nn.Module):
    """
        CSP Spatial Pyramid Pooling Block
    """
    def __init__(self, c1, c2, K1,D1,K2,D2,K3,D3,K4,D4,K5,D5,K6,D6,K7,D7,e=0.5, kernel_sizes=[5, 9, 13]):
        super(SPPBlockCSP, self).__init__()
        self.cv1 = Conv(c1, c1//2, k=1,K=K1,D=D1)
        self.cv2 = Conv(c1, c1//2, k=1,K=K2,D=D2)
        
        self.l1_1 = Conv(c1//2, c1//2, k=3, p=1,K=K3,D=D3)
        self.l1_2 = SPP(c1//2, c1//2, K4,D4,K5,D5,e=e, kernel_sizes=kernel_sizes)
        self.l1_3 = Conv(c1//2, c1//2, k=3, p=1,K=K6,D=D6)
        
        self.cv3 = Conv(c1, c2, k=1, K=K7,D=D7)

        
    def forward(self, x):
        x1,c_spikes1 = self.cv1(x)
        x2,c_spikes2 = self.cv2(x)
        x3,c_spikes3 = self.l1_1(x2)
        x3,[c_spikes4,c_spikes5] = self.l1_2(x3)
        x3,c_spikes6 = self.l1_3(x3)
        
        
        y,c_spikes7 = self.cv3(torch.cat([x1, x3], dim=1))

        return y,[c_spikes1,c_spikes2,c_spikes3,c_spikes4,c_spikes5,c_spikes6,c_spikes7]

class UpSample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(UpSample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corner = align_corner

    def forward(self, x):
        return torch.nn.functional.interpolate(input=x, 
                                               size=self.size, 
                                               scale_factor=self.scale_factor, 
                                               mode=self.mode, 
                                               align_corners=self.align_corner
                                               )
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, K1,D1,K2,D2,shortcut=True, d=1, e=0.5, depthwise=False):
      # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels            
        self.cv1 = Conv(c1, c_, k=1,K=K1,D=D1)
        self.cv2 = Conv(c_, c2, k=3, p=d,K=K2,D=D2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y,c_spikes1 = self.cv1(x)
        y,c_spikes2 = self.cv2(y)
        return y,[c_spikes1,c_spikes2]
        
        
        
class BottleneckCSP(nn.Module):
    def __init__(self, c1, c2,K1,D1,K2,D2,K3,D3,K4,D4,K5,D5, n=1, shortcut=True, e=0.5, depthwise=False):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1,K=K1,D=D1)
        self.cv2 = Conv(c1, c_, k=1,K=K2,D=D2)
        self.cv3 = Conv(2 * c_, c2, k=1,K=K3,D=D3)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, e=1.0, depthwise=depthwise, act=act) for _ in range(n)])
        self.m = Bottleneck(c_, c_, K4,D4,K5,D5,shortcut, e=1.0, depthwise=depthwise)

    def forward(self, x):
        x1,c_spikes1 = self.cv1(x)
        x2,c_spikes2 = self.cv2(x)
        xm,[c_spikes3,c_spikes4] = self.m(x1)
        x3,c_spikes5 = self.cv3(torch.cat((xm, x2), dim=1))
        return x3,[c_spikes1,c_spikes2,c_spikes3,c_spikes4,c_spikes5]
        
class yolo_tiny_DLN(nn.Module):
    def __init__(self, 
                 cfg=None,
                 device=None, 
                 img_size=640, 
                 num_classes=80, 
                 trainable=False, 
                 conf_thresh=0.001, 
                 nms_thresh=0.60,
                 center_sample=False):
        super(yolo_tiny_DLN, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample

        
        feature_channels = [128, 256, 512]
        self.stride = [8, 16, 32]
        
        backbone_CSPDarknetTiny = CSPDarknetTiny(K[1],D1,K[2],D2,K[3],D3,K[4],D4,K[5],D5,K[6],D6,K[7],D7,K[8],D8,K[9],D9,K[10],D10,K[11],D11,K[12],D12,K[13],D13,K[14],D14,K[15],D15,K[16],D16,K[17],D17,K[18],D18,K[19],D19,K[20],D20,K[21],D21,K[22],D22,K[23],D23,K[24],D24,K[25],D25,K[26],D26,K[27],D27,K[28],D28,K[29],D29,K[30],D30,K[31],D31)
        
        self.backbone = backbone_CSPDarknetTiny
        
        #self.stride = strides
        
        anchor_size = cfg["anchor_size"]
        self.anchor_size = torch.tensor(anchor_size).reshape(len(self.stride), len(anchor_size) // 3, 2).float()
        self.num_anchors = self.anchor_size.size(1)
        c3, c4, c5 = feature_channels

        # build grid cell
        self.grid_cell, self.anchors_wh = self.create_grid(img_size)






        # head
        #self.head_conv_0 = build_neck(model=cfg["neck"], in_ch=c5, out_ch=c5//2)  # 10
        self.head_conv_0 = SPPBlockCSP(c5,c5//2,K[32],D32,K[33],D33,K[34],D34,K[35],D35,K[36],D36,K[37],D37,K[38],D38)
        self.head_upsample_0 = UpSample(scale_factor=2)
        self.head_csp_0 = BottleneckCSP(c4 + c5//2, c4,K[39],D39,K[40],D40,K[41],D41,K[42],D42,K[43],D43, n=1, shortcut=False)

        # P3/8-small
        self.head_conv_1 = Conv(c4, c4//2, k=1,K=K[44],D=D44)  # 14
        self.head_upsample_1 = UpSample(scale_factor=2)
        self.head_csp_1 = BottleneckCSP(c3 + c4//2, c3,K[45],D45,K[46],D46,K[47],D47,K[48],D48,K[49],D49, n=1, shortcut=False)

        # P4/16-medium
        self.head_conv_2 = Conv(c3, c3, k=3, p=1, s=2,K=K[50],D=D50)
        self.head_csp_2 = BottleneckCSP(c3 + c4//2, c4,K[51],D51,K[52],D52,K[53],D53,K[54],D54,K[55],D55, n=1, shortcut=False)

        # P8/32-large
        self.head_conv_3 = Conv(c4, c4, k=3, p=1, s=2,K=K[56],D=D56)
        self.head_csp_3 = BottleneckCSP(c4 + c5//2, c5,K[57],D57,K[58],D58,K[59],D59,K[60],D60,K[61],D61, n=1, shortcut=False)

        # det conv

        self.head_det_1 = Conv0(c3, self.num_anchors * (1 + self.num_classes + 4), k=1, K=K[62],D=D62)
        self.head_det_2 = Conv0(c4, self.num_anchors * (1 + self.num_classes + 4), k=1, K=K[63],D=D63)
        self.head_det_3 = Conv0(c5, self.num_anchors * (1 + self.num_classes + 4), k=1, K=K[64],D=D64)



    def init_bias(self):               
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.head_det_1.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.head_det_2.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.head_det_3.bias[..., :self.num_anchors], bias_value)


    def create_grid(self, img_size):
        total_grid_xy = []
        total_anchor_wh = []
        w, h = img_size, img_size
        for ind, s in enumerate(self.stride):
            # generate grid cells
            fmp_w, fmp_h = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            # [HW, 2] -> [1, HW, 1, 2]   
            grid_xy = grid_xy[None, :, None, :].to(self.device)
            # [1, HW, 1, 2]
            anchor_wh = self.anchor_size[ind].repeat(fmp_h*fmp_w, 1, 1).unsqueeze(0).to(self.device)

            total_grid_xy.append(grid_xy)
            total_anchor_wh.append(anchor_wh)

        return total_grid_xy, total_anchor_wh


    def set_grid(self, img_size):
        self.img_size = img_size
        self.grid_cell, self.anchors_wh = self.create_grid(img_size)


    def nms(self, dets, scores):
        """"Pure Python NMS YOLOv4."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds


    @torch.no_grad()
    def inference_single_image(self, x):

        
        
        KA = self.num_anchors
        C = self.num_classes

        t0 = time.time()

        # backbone
        c3, c4, c5,[c_spikes1,c_spikes2,c_spikes3,c_spikes4,c_spikes5,c_spikes6,c_spikes7,c_spikes8,c_spikes9,c_spikes10,c_spikes11,c_spikes12,c_spikes13,c_spikes14,c_spikes15,c_spikes16,c_spikes17,c_spikes18,c_spikes19,c_spikes20,c_spikes21,c_spikes22,c_spikes23,c_spikes24,c_spikes25,c_spikes26,c_spikes27,c_spikes28,c_spikes29,c_spikes30,c_spikes31] = self.backbone(x)

        # FPN + PAN
        # head
        c6,[c_spikes32,c_spikes33,c_spikes34,c_spikes35,c_spikes36,c_spikes37,c_spikes38] = self.head_conv_0(c5)
        c7 = self.head_upsample_0(c6)   # s32->s16
        c8 = torch.cat([c7, c4], dim=1)
        c9,[c_spikes39,c_spikes40,c_spikes41,c_spikes42,c_spikes43] = self.head_csp_0(c8)
        # P3/8
        c10,c_spikes44 = self.head_conv_1(c9)
        c11 = self.head_upsample_1(c10)   # s16->s8
        c12 = torch.cat([c11, c3], dim=1)
        c13,[c_spikes45,c_spikes46,c_spikes47,c_spikes48,c_spikes49] = self.head_csp_1(c12)  # to det
        # p4/16
        c14,c_spikes50 = self.head_conv_2(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16,[c_spikes51,c_spikes52,c_spikes53,c_spikes54,c_spikes55] = self.head_csp_2(c15)  # to det
        # p5/32
        c17,c_spikes56 = self.head_conv_3(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19,[c_spikes57,c_spikes58,c_spikes59,c_spikes60,c_spikes61] = self.head_csp_3(c18)  # to det
        
        


        # det
        pred_s,c_spikes62 = self.head_det_1(c13)
        pred_s = pred_s[0]
        
        pred_m,c_spikes63 = self.head_det_2(c16)
        pred_m = pred_m[0]
        
        pred_l,c_spikes64 = self.head_det_3(c19)
        pred_l = pred_l[0]

        preds = [pred_s, pred_m, pred_l]


        # torch.cuda.synchronize()
        # detect_time = time.time() - t0
        # print('im_detect: {:.3f}s'.format(detect_time))

        obj_pred_list = []
        cls_pred_list = []
        box_pred_list = []
        


        for i, pred in enumerate(preds):
            # [KA*(1 + C + 4), H, W] -> [KA*1, H, W] -> [H, W, KA*1] -> [HW*KA, 1]
            obj_pred_i = pred[:KA, :, :].permute(1, 2, 0).contiguous().view(-1, 1)
            # [KA*(1 + C + 4), H, W] -> [KA*C, H, W] -> [H, W, KA*C] -> [HW*KA, C]
            cls_pred_i = pred[KA:KA*(1+C), :, :].permute(1, 2, 0).contiguous().view(-1, C)
            # [KA*(1 + C + 4), H, W] -> [KA*4, H, W] -> [H, W, KA*4] -> [HW, KA, 4]
            reg_pred_i = pred[KA*(1+C):, :, :].permute(1, 2, 0).contiguous().view(-1, KA, 4)

            
            # txty -> xy
            if self.center_sample:
                xy_pred_i = (reg_pred_i[None, ..., :2].sigmoid() * 2.0 - 1.0 + self.grid_cell[i]) * self.stride[i]
            else:
                xy_pred_i = (reg_pred_i[None, ..., :2].sigmoid() + self.grid_cell[i]) * self.stride[i]

            # twth -> wh
            wh_pred_i = reg_pred_i[None, ..., 2:].exp() * self.anchors_wh[i]
            # xywh -> x1y1x2y2           
            x1y1_pred_i = xy_pred_i - wh_pred_i * 0.5
            x2y2_pred_i = xy_pred_i + wh_pred_i * 0.5
            box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1)[0].view(-1, 4)

            obj_pred_list.append(obj_pred_i)
            cls_pred_list.append(cls_pred_i)
            box_pred_list.append(box_pred_i)
        
        obj_pred = torch.cat(obj_pred_list, dim=0)
        cls_pred = torch.cat(cls_pred_list, dim=0)
        box_pred = torch.cat(box_pred_list, dim=0)
        
        # normalize bbox
        bboxes = torch.clamp(box_pred / self.img_size, 0., 1.)

        # scores
        scores = torch.sigmoid(obj_pred) * torch.softmax(cls_pred, dim=-1)

        # to cpu
        scores = scores.detach().to('cpu').numpy()
        bboxes = bboxes.detach().to('cpu').numpy()

        # post-process
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

        return bboxes, scores, cls_inds,0


    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:

            gtimg = x.clone().detach()
            B = x.size(0)
            KA = self.num_anchors
            C = self.num_classes

            # backbone
            c3, c4, c5,_ = self.backbone(x)

            # FPN + PAN
            # head
            c6,_ = self.head_conv_0(c5)
            c7 = self.head_upsample_0(c6)   # s32->s16
            c8 = torch.cat([c7, c4], dim=1)
            c9,_ = self.head_csp_0(c8)
            # P3/8
            c10,_ = self.head_conv_1(c9)
            c11 = self.head_upsample_1(c10)   # s16->s8
            c12 = torch.cat([c11, c3], dim=1)
            c13,_ = self.head_csp_1(c12)  # to det
            # p4/16
            c14,_ = self.head_conv_2(c13)
            c15 = torch.cat([c14, c10], dim=1)
            c16,_ = self.head_csp_2(c15)  # to det
            # p5/32
            c17,_ = self.head_conv_3(c16)
            c18 = torch.cat([c17, c6], dim=1)
            c19,_ = self.head_csp_3(c18)  # to det

            # det
            pred_s,_ = self.head_det_1(c13)
            pred_m,_ = self.head_det_2(c16)
            pred_l,_ = self.head_det_3(c19)


            preds = [pred_s, pred_m, pred_l]
            obj_pred_list = []
            cls_pred_list = []
            box_pred_list = []
            


            for i, pred in enumerate(preds):
                # [B, KA*(1 + C + 4), H, W] -> [B, KA, H, W] -> [B, H, W, KA] ->  [B, HW*KA, 1]
                obj_pred_i = pred[:, :KA, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
                # [B, KA*(1 + C + 4), H, W] -> [B, KA*C, H, W] -> [B, H, W, KA*C] -> [B, H*W*KA, C]
                cls_pred_i = pred[:, KA:KA*(1+C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
                # [B, KA*(1 + C + 4), H, W] -> [B, KA*4, H, W] -> [B, H, W, KA*4] -> [B, HW, KA, 4]
                reg_pred_i = pred[:, KA*(1+C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, KA, 4)
                # txty -> xy
                if self.center_sample:
                    xy_pred_i = (reg_pred_i[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_cell[i]) * self.stride[i]
                else:
                    xy_pred_i = (reg_pred_i[..., :2].sigmoid() + self.grid_cell[i]) * self.stride[i]
                # twth -> wh
                wh_pred_i = reg_pred_i[..., 2:].exp() * self.anchors_wh[i]
                # xywh -> x1y1x2y2
                x1y1_pred_i = xy_pred_i - wh_pred_i * 0.5
                x2y2_pred_i = xy_pred_i + wh_pred_i * 0.5
                box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1).view(B, -1, 4)

                obj_pred_list.append(obj_pred_i)
                cls_pred_list.append(cls_pred_i)
                box_pred_list.append(box_pred_i)
            
            obj_pred = torch.cat(obj_pred_list, dim=1)
            cls_pred = torch.cat(cls_pred_list, dim=1)
            box_pred = torch.cat(box_pred_list, dim=1)
            
            # normalize bbox
            box_pred = box_pred / self.img_size

            # compute giou between prediction bbox and target bbox
            x1y1x2y2_pred = box_pred.view(-1, 4)
            x1y1x2y2_gt = targets[..., 2:6].view(-1, 4)
            

            gt_densitys = torch.zeros_like(x1y1x2y2_gt[:,0])
            pre_densitys = torch.zeros_like(x1y1x2y2_gt[:,0])
            
            
            
            at_least_one_positive = (x1y1x2y2_gt > 0).any(dim=1)
            #filtered_boxes = x1y1x2y2_gt[at_least_one_positive]

            box_indices = torch.where(at_least_one_positive)[0].cpu().numpy()
            img_indices = box_indices//(80*80*3+40*40*3+20*20*3)

            for i in range (len(box_indices)):
                gtimg0 = gtimg[img_indices[i],2,:,:]

                
                gtbox = x1y1x2y2_gt[box_indices[i]].clone().detach()
                gtbox[gtbox<0]=0
                x1, y1, x2, y2 = (gtbox * 640).to(torch.int32)
                
                img = gtimg0[y1:y2, x1:x2]
                non_zero_count = (img != 0).sum()
                gt_densitys[box_indices[i]] = non_zero_count/((x2-x1)*(y2-y1))
                
                
                pre_box = x1y1x2y2_pred[box_indices[i]].clone().detach()
                pre_box[pre_box<0]=0
                x1, y1, x2, y2 = (pre_box * 640).to(torch.int32)
                
                img_pred = gtimg0[y1:y2, x1:x2]
                non_zero_count_pred = (img_pred != 0).sum()
                pre_densitys[box_indices[i]] = non_zero_count_pred / ((x2 - x1) * (y2 - y1))                

                
            gt_densitys=gt_densitys.view(batch_size, -1)
            pre_densitys=pre_densitys.view(batch_size, -1)

            
            densitys_iou_target = 1.0 - torch.abs(gt_densitys-pre_densitys)
            densitys_iou_target[densitys_iou_target==1.0]=0
            
            densityiou_pred = densitys_iou_target
            densitys_iou_target = densitys_iou_target.unsqueeze(-1)
            # for densitys_iou
            
            
            densitys_iou_target[densitys_iou_target>0]=1.0

            

            ciou_pred = box_ops.ciou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)
            


            
            iou_target = densitys_iou_target
            
            iou_pred = 0.5*densityiou_pred+0.5*ciou_pred



            targets = torch.cat([iou_target, targets], dim=-1)

            
            return obj_pred, cls_pred, iou_pred,  targets


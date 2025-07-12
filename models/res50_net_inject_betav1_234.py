import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .resnet_dilation import resnet50
from .BaseBlocks import BasicConv_PRelu
from .rfb import RFB
from .Injection_v3_beta import Injection
import os

'''
主模型
'''
class conbine_feature(nn.Module):
    def __init__(self, hiddenchannel):
        super(conbine_feature, self).__init__()
        self.hiddenchannel = hiddenchannel
        self.up2_high = DilatedParallelConvBlockD2( self.hiddenchannel, 16) # 32 16
        self.up2_low = nn.Conv2d(256, 16, 1, stride=1, padding=0,bias=False) # res 256 vgg 128
        self.up2_bn2 = nn.BatchNorm2d(16)
        self.up2_act = nn.PReLU(16)
        self.refine=nn.Sequential(nn.Conv2d(16,16,3,padding=1,bias=False),nn.BatchNorm2d(16),nn.PReLU())

    def forward(self, low_fea,high_fea):
        high_fea = self.up2_high(high_fea) # c 16
        low_fea = self.up2_bn2(self.up2_low(low_fea)) # c 16
        refine_feature = self.refine(self.up2_act(high_fea+low_fea)) # 卷积层
        return refine_feature



class DilatedParallelConvBlockD2(nn.Module): # 表面像是降通道的
    def __init__(self, nIn, nOut, add=False):
        super(DilatedParallelConvBlockD2, self).__init__()
        n = int(np.ceil(nOut / 2.)) # 向上取整数
        n2 = nOut - n # 这个不就是减去了一半
        #这里有个问题是既然是降低了，为什么还要按照通道分开，这里没有提到
        self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, bias=False)
        self.conv1 = nn.Conv2d(n, n, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(n2, n2, 3, stride=1, padding=2, dilation=2, bias=False) # 降低了维度

        self.bn = nn.BatchNorm2d(nOut)
        self.add = add
    # 在通道上进行不同的空洞操作类似于八度卷积吗
    def forward(self, input):
        in0 = self.conv0(input) # 先改通道
        in1, in2 = torch.chunk(in0, 2, dim=1) # 按照通道数分块
        b1 = self.conv1(in1) # 空洞率1
        b2 = self.conv2(in2) # 空洞率2
        output = torch.cat([b1, b2], dim=1)

        if self.add:
            output = input + output
        output = self.bn(output)

        return output

class DualFastnet(nn.Module):
    def __init__(self, hiddenchannel=128):  # ,down_factor=4
        super(DualFastnet, self).__init__()
        # num_of_feat = 512
        # 这里是两个encoder
        self.Res50_rgb = resnet50(pretrained=True, output_stride=16, input_channels=3)

        self.hidden = hiddenchannel
        # rfb
        self.tranposelayer_rgb3 = RFB(512,self.hidden)
        self.tranposelayer_rgb4 = RFB(1024,self.hidden)
        self.tranposelayer_rgb5 = RFB(2048,self.hidden)

        # fuse
        self.fuse_fea = BasicConv_PRelu(self.hidden,self.hidden,3,1)
        # decoder b
        self.combine = conbine_feature(self.hidden)
        # Drop这里有什么用
        self.SegNIN = nn.Sequential(nn.Dropout2d(0.1),nn.Conv2d(16, 1, kernel_size=1,bias=False))

        # Injection
        self.inject_2 = Injection(256,256,2,96)
        self.inject_3 = Injection(512,512,3,96)
        self.inject_4 = Injection(1024,1024,4,96)

    def forward(self,rgb,depth):
        # ------------------encoder------------------

        # rgb net
        block0 = self.Res50_rgb.conv1(rgb)
        block0 = self.Res50_rgb.bn1(block0)
        block0 = self.Res50_rgb.relu(block0)  # 256x256
        block0 = self.Res50_rgb.maxpool(block0)  # 128x128
        frist_rgb = self.Res50_rgb.layer1(block0)  # 64x64
        # Injection
        conv2_rgb_inject = self.inject_2(frist_rgb, depth)
        conv3_rgb = self.Res50_rgb.layer2(conv2_rgb_inject + frist_rgb)  # 32x32
        # Injection
        conv3_rgb_inject = self.inject_3(conv3_rgb,depth)
        conv4_rgb = self.Res50_rgb.layer3(conv3_rgb_inject + conv3_rgb)  # 16x16
        # Injection
        conv4_rgb_inject = self.inject_4(conv4_rgb,depth)
        conv5_rgb = self.Res50_rgb.layer4(conv4_rgb_inject + conv4_rgb)  # 8x8

        # ------------------decoder------------------

        # transpose
        conv3_rgb_new = self.tranposelayer_rgb3(conv3_rgb)
        conv4_rgb_new = self.tranposelayer_rgb4(conv4_rgb)
        conv5_rgb_new = self.tranposelayer_rgb5(conv5_rgb)

        # 上采成一个大小
        conv5_rgb_up = F.interpolate(conv5_rgb_new, size=(conv3_rgb.shape[-2],conv3_rgb.shape[-1]))
        conv4_rgb_up = F.interpolate(conv4_rgb_new, size=(conv3_rgb.shape[-2],conv3_rgb.shape[-1]))
        conv5_fused = conv3_rgb_new + conv4_rgb_up + conv5_rgb_up
        conv5_final = self.fuse_fea(conv5_fused)
        #上采样
        # rgb decoder
        rgb_final = F.interpolate(conv5_final, size=(frist_rgb.shape[-2], frist_rgb.shape[-1]),
                                  mode="bilinear",
                                  align_corners=False)
        rgb_final = self.combine(frist_rgb,rgb_final) # 1/8

        rgb_final = F.interpolate(self.SegNIN(rgb_final), size=(rgb.shape[-2], rgb.shape[-1]), mode="bilinear",align_corners=False)

        return rgb_final

if __name__=="__main__":
    # from torchstat import stat
    a=torch.zeros(1,3,256,256).cuda()
    b=torch.zeros(1,3,256,256).cuda()

    mobile=DualFastnet().cuda()

    c =mobile(a, b)
    print(c.size())

    total_paramters = sum([np.prod(p.size()) for p in mobile.parameters()])
    print('Total network parameters: ' + str(total_paramters/1e6)+"M")

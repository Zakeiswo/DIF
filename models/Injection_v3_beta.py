import torch
import torch.nn as nn
import numpy as np
from .BaseBlocks import BasicConv_PRelu
from .denselayer import DenseLayer
from .dysp import dyspatial_module
'''
添加了更多的卷积，再dy后面都加上3x3卷积
'''
class Injection(nn.Module):
    def __init__(self,in_channel, out_channel, layer_num, hidden_channels =32):
        '''
        :param in_channel: 输入通道数对齐layerX
        :param out_channel: 输出通道数
        :param layer_num: 第几层
        :param hidden_channels: 内部的通道数
        '''
        super(Injection, self).__init__()
        # channels
        scales = {0:1,1: 2, 2: 4, 3: 8, 4: 16, 5: 32}
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_channels = hidden_channels
        self.layer_num = layer_num
        self.scale = scales[self.layer_num]
        # dense layer

        self.dense_1 = DenseLayer(self.hidden_channels,self.hidden_channels)
        self.dense_2 = DenseLayer(self.hidden_channels,self.hidden_channels)
        # transpose
        self.transpose_depth = BasicConv_PRelu(3, 16, kernel_size=1)

        self.transpose_shuffle = BasicConv_PRelu(16 * self.scale* self.scale,self.hidden_channels,1)
        self.transpose_down = BasicConv_PRelu(self.in_channel,self.hidden_channels,1)


        self.transpose_up = BasicConv_PRelu(self.hidden_channels, self.out_channel,1)

        # normal conv
        self.shuffle_pre = BasicConv_PRelu(16, 16, kernel_size=3,padding=1)
        self.shuffle_post = BasicConv_PRelu(self.hidden_channels, self.hidden_channels, kernel_size=3,padding=1)
        self.fuse_1 = BasicConv_PRelu(self.hidden_channels, self.hidden_channels,kernel_size=3,padding=1)
        self.fuse_2 = BasicConv_PRelu(self.hidden_channels, self.hidden_channels,kernel_size=3,padding=1)

        # dy conv
        self.dysp = dyspatial_module(self.hidden_channels,self.hidden_channels,self.hidden_channels)
        self.dysp_dp = dyspatial_module(self.hidden_channels,self.hidden_channels,self.hidden_channels)
        # pixelUnshuffle
        self.pixel_unshuffle = nn.PixelUnshuffle(self.scale)

    def forward(self, layerX, depth):
        # N, C, H, W = layerX.size()
        # down
        layerX_down = self.transpose_down(layerX)

        # depth
        depth_up = self.transpose_depth(depth)
        depth_up = self.shuffle_pre(depth_up)
        
        # 检查并调整尺寸，确保高度和宽度是 scale 的倍数
        # _, _, h, w = depth_up.size()
        # pad_h = (self.scale - h % self.scale) % self.scale
        # pad_w = (self.scale - w % self.scale) % self.scale
        # if pad_h > 0 or pad_w > 0:
        #     depth_up = torch.nn.functional.pad(depth_up, (0, pad_w, 0, pad_h))
        
        # pixelshuffle
        depth_up = self.pixel_unshuffle(depth_up)
        depth_resize = self.transpose_shuffle(depth_up)
        depth_resize = self.shuffle_post(depth_resize)
        # stage 1
        fuse_fea = self.dense_1(depth_resize + layerX_down)  # 我觉得应该在dense之前加
        depth_fine = self.dysp_dp(depth_resize, fuse_fea)
        depth_fine = self.fuse_1(depth_fine)
        # stage 2
        fuse_fea_2 = self.dense_2(depth_fine + layerX_down)
        layerX_fine = self.dysp(layerX_down, fuse_fea_2)
        layerX_fine = self.fuse_2(layerX_fine)
        # up
        out_fea = self.transpose_up(layerX_fine)

        return out_fea

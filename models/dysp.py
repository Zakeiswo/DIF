import torch
import torch.nn as nn
from .BaseBlocks import BasicConv_PRelu
class dyspatial_module(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, down_factor=1):
        """DepthDC3x3_1，利用nn.Unfold实现的动态卷积模块
        这里的x应该是被卷的，y是核
        Args:
            in_xC (int): 第一个输入的通道数 rgb
            in_yC (int): 第二个输入的通道数 kernel
            out_C (int): 最终输出的通道数
            down_factor (int): 用来降低卷积核生成过程中的参数量的一个降低通道数的参数

            这个版本改为是通道的和空间的平行，并且采用分组的方式，最后cat在一起然后洗牌
        """
        super(dyspatial_module, self).__init__()

        self.kernel_size = 3
        # self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.fuse = BasicConv_PRelu(in_yC, out_C, 3, 1, 1)

        self.gernerate_kernel_spatial = nn.Sequential(
            # nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            BasicConv_PRelu(in_yC, in_yC, 3, 1, 1),
            # DenseLayer(in_yC, in_yC, k=down_factor),
            BasicConv_PRelu(in_yC, self.kernel_size ** 2, 1),# in_xC
            #N C W H -> N k2 W H
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)
        self.padding = 1
        self.dilation = 1
        self.stride = 1
        self.dynamic_bias = None

    def forward(self, x, y):  # x : focal y :kernel from image
        # 这里的rgb：batch 32 h w， focal：（12*batch）32 h w
        # 这里可以添加一个xy通道数不一样就报错的功能

        xN, xC, xH, xW = x.size() # focal：（12*batch）32 h w
        yN, yC, yH, yW = y.size()  # rgb：batch 32 h w

        # spatial filter
        kernel = self.gernerate_kernel_spatial(y)  # rgb：batch 32 h w -> batch k2 h w
        kernel = kernel.reshape([yN, self.kernel_size ** 2, yH, yW, 1])  # batch k2 h w -> batch k2 h w 1

        # 这里就应该是kernel的部分
        kernel = kernel.permute(0, 2, 3, 1, 4).contiguous()  # N H W k2 1


        unfold_x = self.unfold(x).reshape([yN, yH, yW, yC, -1])  # N H W C k2
        # 这里就是两个矩阵的低维度相乘
        spatial_after = torch.matmul(unfold_x, kernel)  # N H W k2 1×N H W C k2 = N H W C 1
        spatial_after = spatial_after.squeeze(4).permute(0, 3, 1, 2)  # N C H W

        return spatial_after

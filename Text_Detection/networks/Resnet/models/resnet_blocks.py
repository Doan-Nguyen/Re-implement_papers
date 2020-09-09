import torch
import torch.nn as nn

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels= in_channels,
        out_channels= out_channels,
        kernel_size=1,
        stride= stride, 
        bias=False)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels= in_channels,
        out_channels= out_channels,
        kernel_size=3,
        stride= stride,
        padding=1,
        bias= False)

def residual_blocks(in_channels, out_channels):
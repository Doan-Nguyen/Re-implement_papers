import torch
import torch.nn as nn 
from torch import optim
import torch.nn.functional as F 


### Constracting path 
class DoubleConv(nn.Module):
    """ A path of Contracting path
    2 x [Convolutions (3x3), padding=1 -> ReLU -> MaxPooling (stride=2)]
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        ### architecture
        if mid_channels:
            mid_channels = out_channels
        self.down_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.down_conv(x)


class DownSamplingBlock(nn.Module):
    """ Downscaling with maxpool 
    [max_pool -> ConvsBlock]
    """
    def __init__(self, in_channels, out_channels):
        super(DownSamplingBlock, self).__init__()
        ###
        self.down_sampling = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DownSamplingBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_sampling(x)


### Expansive path
class UpSamplingBlock(nn.Module):
    """ A path of expansive path
    [up-conv (2x2) -> concatenation -> 2 conv (3x3) -> relu]
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpSamplingBlock, self).__init__()
        ### architecture
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode='bilinear', 
                align_corners=True)
            self.conv = DoubleConv(
                in_channels= in_channels, 
                out_channels= out_channels,
                mid_channels= in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels= in_channels,
                out_channels= in_channels // 2, kernel_size=2, 
                stride=2)
            self.conv = DoubleConv(
                in_channels= in_channels,
                out_channels= out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        ## input is CHW
        diff_Y = x2.size()[2] - x1.size()[2]
        diff_X = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_X // 2, diff_X - (diff_X // 2),
                        diff_Y // 2, diff_Y - (diff_Y // 2)])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
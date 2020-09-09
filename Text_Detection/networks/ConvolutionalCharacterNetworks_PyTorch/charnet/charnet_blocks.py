"""
This files to define CharNets's blocks. Including:
    - Residual block
    - Hourglass Module
"""

from imports import *


class ResidualBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResidualBlock, self).__init__()
        """
        convolution (3x3) -> relu
        The input & output channel number is channel_num
        """
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            nn.BatchNorm2d(num_features=channel_num),
            nn.ReLU(inplace=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x 
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        output = self.relu(x)

        return output
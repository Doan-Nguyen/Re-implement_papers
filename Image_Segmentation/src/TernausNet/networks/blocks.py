import torch
import torch.nn as nn 
from torch import optim
from torchvision import models
import torch.nn.functional as F 


def conv3x3(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size=3, padding=1)


class ConvRelu(nn.Module):
    """     The path Conv(3x3)-ReLU in decoder path.     """
    def __init__(self, in_c: int, out_c: int) -> None:
        super(ConvRelu, self).__init__()
        ##   Modify architecture
        self.conv = conv3x3(in_c, out_c)
        self.activattion = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activattion(x)
        return x


class Decoder(nn.Module):
    """     The decoder path, includes:
    - [Conv(3x3) -> ReLU] ~ conv3x3(in_channels, mid_channels)
    - ConvTranspose2d(3x3, stride=2) -> ReLU
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super(Decoder, self).__init__()
        ###
        self.decoder_block = nn.Sequential(
            conv3x3(in_channels= in_channels, out_channels= mid_channels),
            nn.ConvTranspose2d(
                in_channels=mid_channels, 
                out_channels= out_channels, 
                kernel_size=3, 
                stride=2,
            ),
            nn.ReLU(inplace=True),
        ) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder_block(x)
        return x
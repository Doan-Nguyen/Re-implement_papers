import torch
import torch.nn as nn 
from torch import optim
from torchvision import models
import torch.nn.functional as F 


def conv3x3(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size=3, padding=1)


class ConvRelu(nn.Module):
    """     The path Conv-ReLU in decoder path.     """
    def __init__(self, in_c: int, out_c: int) -> None:
        super(self, ConvRelu).__init__()
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
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
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


class UNet_VGG11(nn.Module):
    def __init__(self, num_filters: int = 32, pretrained: bool = False) -> None:
        """     Model using VGG-11 as an encoder. 
        VGG-11/16's architecture can check in "sequential_vgg.md"
        Args:
            - num_filters:
            - pretrained: 
        """
        super(UNet_Vgg11, self).__init__()
        ###
        self.pool = nn.MaxPool2d(kernel_size=2)
        ###          Encoder path
        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = self.encoder[1]     # take one relu layer
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3_1 = self.encoder[6]
        self.conv3_2 = self.encoder[8]
        self.conv4_1 = self.encoder[11]
        self.conv4_2 = self.encoder[13]
        self.conv5_1 = self.encoder[16]
        self.conv5_2 = self.encoder[18]

        ###         Center path
        self.center = Decoder(
            in_channels=512, mid_channels= 512, out_channels=256
        )

        ###         Decoder path
        self.decoder_5 = Decoder(
            in_channels=256 + 512, mid_channels= 512, out_channels= 256
        )
        self.decoder_4 = Decoder(
            in_channels= 256 + 512, mid_channels= 512, out_channels= 128
        )
        self.decoder_3 = Decoder(
            in_channels= 128 + 256, mid_channels= 256, out_channels= 64
        )
        self.decoder_2 = Decoder(
            in_channels= 64 + 128, mid_channels= 128, out_channels= 32
        )
        self.decoder_1 = ConvRelu(in_c= 32 + 64, out_c= 32) ## ??? 

        self.final = nn.Conv2d(32, out_channels= 1, kernel_size=1)

    def forward(self, x):
        ###         Encoder path
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(x))
        conv3_1 = self.relu(self.conv3_1(x))
        conv3_2 = self.relu(self.conv3_2(x))
        conv4_1 = self.relu(self.conv4_1(x))
        conv4_2 = self.relu(self.conv4_2(x))
        conv5_1 = self.relu(self.conv5_1(x))
        conv5_2 = self.relu(self.conv5_2(x))
        ###         Center path
        center = self.relu(self.center(x))
        ###         Decoder path

        decoder5 = self.decoder_5(torch.cat([conv5_2, center], 1))
        decoder4 = self.decoder_4(torch.cat([decoder5, conv4_2], 1))
        decoder3 = self.decoder_3(torch.cat([decoder4, conv3_2], 1))
        decoder2 = self.decoder_2(torch.cat([decoder3, conv2], 1))
        decoder1 = self.decoder_1(torch.cat([decoder2, conv1], 1))
        
        return self.final(decoder1) 










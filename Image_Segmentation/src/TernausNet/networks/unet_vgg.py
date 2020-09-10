import logging
import os 
import numpy as np 
import sys

import torch
import torch.nn as nn 
from torch import optim
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter

# import blocks
from .blocks import *



class UNet_VGG11(nn.Module):
    def __init__(self, num_filters: int = 32, pretrained: bool = False) -> None:
        """     Model using VGG-11 as an encoder. 
        VGG-11/16's architecture can check in "sequential_vgg.md"
        Args:
            - num_filters:
            - pretrained: 
        """
        super(UNet_VGG11, self).__init__()
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


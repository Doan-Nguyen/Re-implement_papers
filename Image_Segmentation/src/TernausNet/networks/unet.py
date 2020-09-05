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


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        """
        DownSamplingBlock: 64 -> 128 -> 256 -> 512
        """
        
        super(UNet, self).__init__()
        ###
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(
                in_channels=n_channels,
                out_channels=64, 
                mid_channels= None)
        self.down_1 = DownSamplingBlock(
                in_channels= 64, 
                out_channels= 128)
        self.down_2 = DownSamplingBlock(
                in_channels= 128, 
                out_channels= 256)
        self.down_3 = DownSamplingBlock(
                in_channels= 256, 
                out_channels= 512)
        if bilinear:
            factor = 2
        else:
            factor = 1
        self.down_4 = DownSamplingBlock(
                in_channels=512, 
                out_channels= 1024 // factor)
        self.up_1 = UpSamplingBlock(
                in_channels=1024,
                out_channels= 512 //factor,
                bilinear= bilinear)
        self.up_2 = UpSamplingBlock(
                in_channels=512, 
                out_channels= 256 // factor, 
                bilinear= bilinear)
        self.up_3 = UpSamplingBlock(
                in_channels=256, 
                out_channels= 128 // factor, 
                bilinear= bilinear)
        self.up_4 = UpSamplingBlock(
                in_channels=128, 
                out_channels= 64, 
                bilinear= bilinear)
        self.out = OutConv(
                in_channels=64,
                out_channels= n_classes) 

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

        


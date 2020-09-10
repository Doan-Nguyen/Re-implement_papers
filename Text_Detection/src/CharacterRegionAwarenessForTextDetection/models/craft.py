# -*- coding: utf-8 -*-
import logging

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from vgg_bn import init_weights, VGG16_BN

###         Logging for debug model
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.debug("This is architecture logging")


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        """
        Architecture:
            [conv2d -> BN -> ReLU] *2
        """
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True) 
        )
    
    def forward(self, x):
        return self.double_conv(x)

    
class CRAFT(nn.Module):
    def __init__(self, pretrained=True, freeze=False):
        super(CRAFT, self).__init__()
        ###
        self.basenet = VGG16_BN(pretrained, freeze=False)
        #           U-net
        self.upconv1 = DoubleConv(in_ch=1024, mid_ch=512, out_ch=256)  ## bug ?
        self.upconv2 = DoubleConv(in_ch=512, mid_ch=256, out_ch=128)
        self.upconv3 = DoubleConv(in_ch=256, mid_ch=128, out_ch=64)
        self.upconv4 = DoubleConv(in_ch=128, mid_ch=64, out_ch=32)

        num_class = 2
        self.conv_classifier = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),  nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),  nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),  nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),  nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=num_class, kernel_size=1)
        ) 

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_classifier.modules())

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        # ToDo - Remove the interpolation and make changes in the dataloader to make target width, height //2

        y = F.interpolate(y, size=(768, 768), mode='bilinear', align_corners=False)

        return y

if __name__ == '__main__':
    model = CRAFT(pretrained=True).cuda()
    output, _ = model(torch.randn(1, 3, 768, 768).cuda())
    print(output.shape)



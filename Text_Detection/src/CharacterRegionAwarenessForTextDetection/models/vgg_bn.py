# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from torchvision.models.vgg import model_urls


def init_weights(modules):
    """

    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class VGG16_BN(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(VGG16_BN, self).__init__()
        ###
        model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
        #   Details vgg16_bn's architecture at vgg16_bn.md
        vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features
        #   Take subpath in vgg16_bn
        self.slice1 = torch.nn.Sequential()
        for x in range(12):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        self.slice2 = torch.nn.Sequential()
        for x in range(12, 19):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        self.slice3 = torch.nn.Sequential()
        for x in range(19, 29):
            self.slice3.add_module(str(x))

        self.slice4 = torch.nn.Sequential()
        for x in range(29, 39):
            self.slice3.add_module(str(x))
        ## fc6, fc7
        self.slice5 = torch.nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6),
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())
        init_weights(self.slice5.modules())

        if freeze:
            for param in self.slice1.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out






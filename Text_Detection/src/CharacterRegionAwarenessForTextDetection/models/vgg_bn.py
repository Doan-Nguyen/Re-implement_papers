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
            init.xavier_uniform_(m.weight.data: tensor)
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
            self.slice1.add_module(str(x): str, vgg_pretrained_features[x]: 'Module')

        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

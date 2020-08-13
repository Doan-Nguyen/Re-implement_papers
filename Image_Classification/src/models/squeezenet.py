"""squeezenet in pytorch
[1] Song Han, Jeff Pool, John Tran, William J. Dally

    squeezenet: Learning both Weights and Connections for Efficient Neural Networks
    https://arxiv.org/abs/1506.02626
"""
import torch
import torch.nn as nn


class FireModule(nn.Module):
    """             Build Fire Module
    Squeeze_path:
        - 3 {conv (1x1)},  ReLu
    Expand path:
        - 4 {conv(1x1) ; conv(3x3)}, ReLu
    """
    def __init__(self, in_channels, squeeze_channels, exp11_channels, exp33_channels):
        """
        Parameters:
            - in_channels:
            - out_channels:
            - squeeze_channels:
        """
        super(FireModule, self).__init__()
        ###
        self.squeeze_11 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.expand_11 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_channels, out_channels=exp11_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.expand_33 = nn.Sequential(
            nn.Conv2d(in_channels=exp11_channels, out_channels=exp33_channels, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze_11(x)
        x = torch.cat([self.expand_11(x), self.expand_33(x)], dim=1)  # horizontal
        return x

class SqueezeNetSimplePass(nn.Module):
    """SqueezeNet with simple bypass"""

    def __init__(self, num_classes=1000):
        super(SqueezeNetSimplePass, self).__init__()
        self.num_classes = num_classes
        ###     model architecture flows 'Table 1: SqueezeNet architectural dimensions'
        self.features = nn.Sequential(
            # input convolution
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # fire module
            FireModule(in_channels=96, squeeze_channels=16, exp11_channels=64, exp33_channels=64),  # fire2
            FireModule(in_channels=128, squeeze_channels=16, exp11_channels=64, exp33_channels=64),  # fire3
            FireModule(in_channels=128, squeeze_channels=32, exp11_channels=128, exp33_channels=128),  # fire4
            nn.MaxPool2d(kernel_size=3, stride=2),
            FireModule(in_channels=256, squeeze_channels=32, exp11_channels=128, exp33_channels=128),  # fire5
            FireModule(in_channels=256, squeeze_channels=48, exp11_channels=192, exp33_channels=192),  # fire6
            FireModule(in_channels=384, squeeze_channels=48, exp11_channels=192, exp33_channels=192),  # fire7
            FireModule(in_channels=384, squeeze_channels=64, exp11_channels=256, exp33_channels=256),  # fire8
            nn.MaxPool2d(kernel_size=3, stride=2),
            FireModule(in_channels=512, squeeze_channels=64, exp11_channels=256, exp33_channels=256),  # fire9  
        )
        self.classifier = nn.Sequential(
            # output convolution: out_channels = numb_classes
            nn.Conv2d(in_channels=512, out_channels=self.num_classes, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),  
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def squeezenet(num_classes=47):
    return SqueezeNetSimplePass(num_classes=num_classes)


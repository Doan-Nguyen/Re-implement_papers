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
        self.expand__33 = nn.Sequential(
            nn.Conv2d(in_channels=exp11_channels, out_channels=exp33_channels, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = squeeze_11(x)
        x = expand_11(x)
        x = expand_33(x)

        return x

class SqueezeNet(nn.Module):

    """mobile net with simple bypass"""
    def __init__(self, num_class=58):

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fire2 = Fire(96, 128, 16)
        self.fire3 = Fire(128, 128, 16)
        self.fire4 = Fire(128, 256, 32)
        self.fire5 = Fire(256, 256, 32)
        self.fire6 = Fire(256, 384, 48)
        self.fire7 = Fire(384, 384, 48)
        self.fire8 = Fire(384, 512, 64)
        self.fire9 = Fire(512, 512, 64)

        self.conv10 = nn.Conv2d(512, num_class, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)
            
    def forward(self, x):
        x = self.stem(x)

        f2 = self.fire2(x)
        f3 = self.fire3(f2) + f2
        f4 = self.fire4(f3)
        f4 = self.maxpool(f4)

        f5 = self.fire5(f4) + f4
        f6 = self.fire6(f5)
        f7 = self.fire7(f6) + f6
        f8 = self.fire8(f7)
        f8 = self.maxpool(f8)

        f9 = self.fire9(f8)
        c10 = self.conv10(f9)

        x = self.avg(c10)
        x = x.view(x.size(0), -1)

        return x

def squeezenet(num_classes=47):
    return SqueezeNet(num_class=num_classes)


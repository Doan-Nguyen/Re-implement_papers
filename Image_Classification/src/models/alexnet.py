import torch
import torch.nn as nn 


class AlexNet(nn.Module):
    def __init__(self, in_channels=224):
        """
        Recommend:  
            - [Conv2d -> ReLU -> MaxPool2d]
            - [Conv2d -> ReLU -> Conv2d]
        """
        super(AlexNet, self).__init__()
        ###
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifies = 
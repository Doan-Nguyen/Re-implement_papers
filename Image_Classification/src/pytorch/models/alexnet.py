import torch
import torch.nn as nn 


class AlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        """
        Recommend:  
            - [Conv2d -> ReLU -> MaxPool2d]
            - [Conv2d -> ReLU -> Conv2d]
        """
        super(AlexNet, self).__init__()
        ###
        self.features = nn.Sequential(
            #
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.classifies = nn.Sequential(
            nn.Linear(6400, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifies(x)
        return x 
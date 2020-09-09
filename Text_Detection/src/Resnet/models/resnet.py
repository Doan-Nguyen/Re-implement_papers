import torch 
import torch.nn as nn 



class ResNet(nn.Module):
    def __init__(self, in_channels):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(
                                in_channels= in_channels,
                                out_channels=112, 
                                kernel_size= 7,
                                stride= 2
                                )
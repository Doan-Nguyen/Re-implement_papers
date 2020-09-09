import torch
import torch.nn as nn

class BasicConvBlock(nn.Module):
    def __init__(self, 
                in_channels, 
                out_channels, 
                kernel_size, 
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                relu=True,
                batch_norm=True,
                bias=True):
        """
        Parameters:
            - in_channels:
            - out_channels:
            - dilation:
            - groups:
            - bn: 
        """
        super(BasicConvBlock, self).__init__()
        self.out_channels = out_channels, 
        self.conv = nn.Conv2d(in_channels, 
                            out_channels, 
                            kernel_size= kernel_size, 
                            stride= stride, 
                            padding= padding,
                            dilation= dilation, 
                            groups= groups,
                            bias= bias)
        self.batch_norm = nn.BatchNorm2d(
                            out_channels, 
                            eps=1e-5, 
                            momentum=0.1, 
                            affine=True) if batch_norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        
        def forward(self, x):
            x = self.conv(x)
            if self.batch_norm is not None:
                x = self.batch_norƯm(x)
            if self.relu is not None:
                x = self.relu(x)
            
            return x

            
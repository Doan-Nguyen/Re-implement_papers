import torch
import torch.nn as nn 


class PrintShape(nn.Module):
    def __init__(self):
        super(PrintShape, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x 


class DebugGradien(nn.Module):
    """
    https://pytorch.org/docs/stable/nn.html?highlight=register_backward_hook#torch.nn.Module.register_backward_hook
    tensor.register_hook
    """
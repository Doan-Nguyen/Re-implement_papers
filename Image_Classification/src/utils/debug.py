import sys

import torch 
from torchvision import models 
from torchs
 

sys.path.append('../models')
import alexnet

def show_information():
    """
    This function to show layer's shape.
    Input tensor X random but same size with input images
    """
    X = torch.rand(1, 1, 224, 224)

    net = alexnet.AlexNet(in_channels=3, num_classes=1000)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'Output shape:\t',X.shape)

# def show_parameters():


if __name__ == "__main__":
    show_information()
import numpy as np
import utils 
from torch import nn
import torch


def get_jaccard(y_true, y_pred):
    """     Phần giao / phần bù

    """
    print("Y true: {}".format(y_true))
    epsilon = 1e-15 
    intersection = (y_pred*y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return list(((intersection + epsilon) / (union - intersection + epsilon)).data.gpu().numpy())
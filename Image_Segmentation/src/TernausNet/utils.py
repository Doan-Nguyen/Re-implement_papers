import json
from datatime import datetime
from pathlib import Path
import random
import numpy as np
import tqdm

import torch


def check_cuda():
    """
    Return:
        gpu_device (int): the number of gpu
    """
    gpu_device = torch.cuda.current_device() 
    return gpu_device

def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))

def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def check_input_size(img_height, img_width):
    """         Check image size (width, height) % 32 == 0         """
    return img_height % 32 == 0 and img_width % 32 == 0


def cyclic_lr(epoch, init_lr=1e-4, num_epochs_per_cycle=5, cycle_epochs_decay=2, lr_decay_factor=0.5):
    epoch_in_cycle = epoch % num_epochs_per_cycle
    lr = init_lr * (lr_decay_factor ** (epoch_in_cycle // cycle_epochs_decay))
    return lr


def batches(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]
##              Basic libaries
import sys
import os
from pathlib import Path
import numpy as np
##              Framework
import tensorflow as tf
##              Files
current_path = Path(os.getcwd())
sys.path.append(str(current_path))
import configs_param
from models.simple_convnet import simple_model
##              Logging


def test_single_image(img_path):
    """         Test predict a image
    Args:
        img_path (str): the image's path
    Return:
        class_score (float): the predict's probability
    """
    model = simple_model() # first_test.ckpt
    print(os.path.join(configs_param.CKPT_DIR, configs_param.MODEL_RESULTS))
    model.load_weights(os.path.join(configs_param.CKPT_DIR, configs_param.MODEL_RESULTS))

    loss, acc= model.evaluate()

    return cls_prob

if __name__ == '__main__':
    # /home/datasets/cifar10_imgs/test_image/0001.png
    prob = test_single_image('/home/datasets/cifar10_imgs/test_image/0001.png')
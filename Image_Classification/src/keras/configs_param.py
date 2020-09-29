##              Basic libaries
import sys
import argparse
import os
from pathlib import Path


current_path = Path(os.getcwd())

CKPT_DIR = os.path.join(current_path, './checkpoints')
MODEL_RESULTS = 'first_test.h5'
TRAIN_DIR = '/home/datasets/cifar10_imgs/train'
VALID_DIR = '/home/datasets/cifar10_imgs/test'
LOG_DIR = './logs/scalars'

EPOCHS = 25
BATCH_SIZE = 16
NUM_CLASSES = int(len(os.listdir(TRAIN_DIR)))


image_height = 32
image_width = 32
channels =3


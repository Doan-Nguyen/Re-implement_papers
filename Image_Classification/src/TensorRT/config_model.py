import argparse 
import os 


# some training parameters
EPOCHS = 50
BATCH_SIZE = 8
NUM_CLASSES = 100
image_height = 224
image_width = 224
channels = 3
model_dir = "image_classification_model.h5"
train_dir = "/media/doannn/data/Projects/Works/SealProject/Datasets/data_from_internet/cifar100_imgs/train"
valid_dir = "/media/doannn/data/Projects/Works/SealProject/Datasets/data_from_internet/cifar100_imgs/test"
# test_dir = "dataset/test"
test_image_path = ""
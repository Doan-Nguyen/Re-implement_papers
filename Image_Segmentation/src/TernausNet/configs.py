import os 
import argparse
from datetime import datetime

###         Path
CHECKPOINT_PATH = './checkpoint'
DATASET_FOLDER = '/media/doannn/Data/Self/Datasets/Bone-Gum_Lines'
TRAIN_FOLDER = os.path.join(DATASET_FOLDER, "imgs_results")
MASK_FOLDER = os.path.join(DATASET_FOLDER, "masks_results")
#   Tensorboard log dir
LOG_DIR = 'runs'


###             Model parameters
EPOCHS = 200
SAVE_EPOCHS = 20      #  save weights file per SAVE_EPOCH epoch
START_EPOCHS = 101   # start epoch for resume training
NEW_EPOCH = 50

#   Mean and std of cifar100 dataset
TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404) 
#  Dataloader
NUMB_CLASSES = 2
NUM_WORKERS = 4

VAL_PERCENT = 10.0
BATCH_SIZE = 32


def get_args():
    """
    args = get_args()
    args.epochs
    """
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ###         Datasets path
    parser.add_argument('-d', '--dataset', dest='dataset', type=str, 
                        default='/media/doannn/Data/Self/Datasets/Bone-Gum_Lines',
                        help='The datasets path')

    ###         Model's parameters
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--val-percent', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()
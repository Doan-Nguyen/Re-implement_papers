import os 
import argparse


num_workers = 4
val_percent = 0.1
batch_size = 4
dataset_path = '/media/doannn/Data/Self/Datasets/Bone-Gum_Lines'
img_path = os.path.join(dataset_path, 'imgs_results')
mask_path = os.path.join(dataset_path, 'masks_results')

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
##              Basic libaries
import sys
import argparse

EPOCHS = 100
BATCH_SIZE = 32
NUM_CLASSES = 10
CKPT_DIR = './checkpoints'
MODEL_RESULTS = 'first_test.h5'
TRAIN_DIR = '/home/datasets/cifar10_imgs/train'
TEST_DIR = '/home/datasets/cifar10_imgs/test'
LOG_DIR = './logs/scalars'

image_height = 32
image_width = 32
channels =3


def parse_args():
    """         Configs the parameters for model            """
    parser = argparse.ArgumentParser(description='Object Classification')

    ###             Paths config
    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                        default="checkpoints/model.ckpt-400000",
                        help='the path of pretrained model to be used', type=str)

    parser.add_argument('--train_dir', dest='train_dir', 
                        default='/media/doannn/data/Projects/Works/SealProject/Datasets/data_from_internet/cifar10_imgs/train',
                        help='The directory where the train files are stored.', type=str)

    parser.add_argument('--test_dir', dest='test_dir', 
                        default='/media/doannn/data/Projects/Works/SealProject/Datasets/data_from_internet/cifar10_imgs/test',
                        help='The directory where the test files are stored.', type=str)

    # parser.add_argument('--crop_dir', dest='crop_dir',
    #                     help='The directory where the crop images are stored.',
    #                     default='../data_ocrpl/results_detect/results_crop', type=str)
    # parser.add_argument('--txt_dir', dest='txt_dir',
    #                     help='The directory where the text files are stored.',
    #                     default='../data_ocrpl/results_detect/results_txt', type=str)
    # parser.add_argument('--visual_dir', dest='visual_dir',
    #                     help='The directory where the visualization results are stored.',
    #                     default='../data_ocrpl/module1_results/visualizations', type=str)

    """                     Config arguments                                """             
    ##          Datasets
    parser.add_argument('--eval_image_width', dest='eval_image_width',
                        help='resized image width for inference',
                        default=1280, type=int)
    parser.add_argument('--eval_image_height', dest='eval_image_height',
                        help='resized image height for inference',
                        default=768, type=int)   
    parser.add_argument('--pixel_conf_threshold', dest='pixel_conf_threshold',
                        help='threshold on the pixel confidence',
                        default=0.5, type=float) 
    parser.add_argument('--link_conf_threshold', dest='link_conf_threshold',
                        help='threshold on the link confidence',
                        default=0.5, type=float) 
    parser.add_argument('--moving_average_decay', dest='moving_average_decay',
                        help='The decay rate of ExponentionalMovingAverage',
                        default=0.9999, type=float)               
    ###         Models
    parser.add_argument('--gpu_memory_fraction', dest='gpu_memory_fraction',
                        help='the gpu memory fraction to be used. If less than 0, allow_growth = True is used.',
                        default=0, type=float)
    parser.add_argument('--batch_size', dest='batch_size', default=16,
                            help='the batch size', type=float)
                              
                        
    global args
    args = parser.parse_args()
    return args


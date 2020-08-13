""" configurations for this project

author baiyu
"""
import os
from datetime import datetime


##          Directory to save weights file & datasets
CHECKPOINT_PATH = './checkpoint'
TRAIN_FOLDER = "/media/doannn/Data/Work/Seal/Sign_Recognition/To_Use_Datasets/train",
TEST_FOLDER = "/media/doannn/Data/Work/Seal/Sign_Recognition/To_Use_Datasets/test",


###             Model parameters

#mean and std of STAMP dataset
TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Dataloader
batch_size = 32 # 48
num_workers = 4

#total training epoches
EPOCH = 101 
MILESTONES = [60, 120, 160]

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().isoformat()

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 20
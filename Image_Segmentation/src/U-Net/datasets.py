# Standard library 
import os 
from os.path import splitext
import numpy as np 
from glob import glob
import logging 
from PIL import Image
import matplotlib.pyplot as plt


import torch
from torch.utils.data import DataLoader, random_split, Dataset
import sys

# local applications 
import configs_param


class BasicDatasets(Dataset):
    """ This class to preprocess datasets to train & for RGB image. Take:
        - __len__(): the number of images (mask images)
        - __getitem__(): load data & get label
        - preprocess() ~ 
    """
    def __init__(self, imgs_dir, masks_dir, scale=1.0):
        """
        Parameters:
            - imgs_dir: the image's path
            - masks_dir: the masks image's path
            - scale: to resize images (0 < scale <= 1)
        """
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale

        assert 0 < scale <= 1, "Scale must be between 0 & 1"
        self.ids = [splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size 
        
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        ### HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        ### take image & mask image by get detail paths
        for image in os.listdir(self.imgs_dir):
            mask_name = image.replace('.jpg', 'png')
            img_path = os.path.join(self.imgs_dir, image)
            mask_path = os.path.join(self.masks_dir, mask_name)

            mask = Image.open(mask_path)
            img = Image.open(img_path)

            assert img.size == mask.size, \
                logging.warning(f'Origin image {img_path} different size with mask images {mask_path}')
                # (f'Origin image {img_path} different size with mask images {mask_path}')

            img = self.preprocess(img, scale=1.0)
            mask = self.preprocess(mask, scale=1.0)

            return {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor)
            }


class CustomDataset(Dataset):
    def __init__(self, image_paths, target_paths):   
        """
        Args:
            - image_paths (str): the path where the images is located
            - target_paths (str): the path where the masks is located
            - 
        """
        # /media/doannn/Data/Self/Datasets/Bone-Gum_Lines/data_dataset_voc_bose/bone/imgs_results
        
        ### all file name
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.image_arr = glob(str(self.image_paths) + "/*")
        self.target_arr = glob(str(self.target_paths) + "/*")

        ### the number of images/masks
        self.data_len = len(self.image_arr)        
    
    def __getitem__(self, i):
        ### take image & mask image by get detail paths
        for image in self.image_arr:
            mask_name = image.replace('.jpg', 'png')
            img_path = os.path.join(self.image_paths, image)
            mask_path = os.path.join(self.target_paths, mask_name)

            mask = Image.open(mask_path)
            img = Image.open(img_path)

            assert img.size == mask.size, \
                logging.warning(f'Origin image {img_path} different size with mask images {mask_path}')
                # (f'Origin image {img_path} different size with mask images {mask_path}')

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }

    def __len__(self):  
        """
        Returns:
            length (int): length of the data
        """
        return self.data_len


def dataloader(imgs_dir, masks_dir):
    """
    Parameters:
        - 
    """
    val_percent = configs_param.val_percent
    batch_size = configs_param.batch_size 

    dataset = CustomDataset(imgs_dir, masks_dir)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    ### data loader
    train_loader = DataLoader(
                    train, 
                    batch_size=batch_size,
                    shuffle= True, 
                    num_workers= configs_param.num_workers,   
                    pin_memory=True)  # load dataset on CPU & push to GPU during training -> speed up
    val_loader = DataLoader(
                    val, 
                    batch_size=batch_size,
                    shuffle= True, 
                    num_workers= configs_param.num_workers, 
                    pin_memory=True, 
                    drop_last= True)

    return train_loader, val_loader, n_train, n_val

def plot_img_and_mask(img, mask):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

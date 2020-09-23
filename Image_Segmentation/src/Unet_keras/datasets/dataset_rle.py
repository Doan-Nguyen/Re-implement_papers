"""         
Datasets includes:
    - file train.csv
    - image folders:
        - original images
            - abc.jpg
            ...
        - mask images
            - abc_mask.png
            ...
"""
#       Basic libaries
import sys
import os
import pandas as pd
import numpy as np
from PIL import Image
#       Files
sys.path.append('../')
import configs_params


def read_csv(csv_path: str):
    """     This function loads .csv files & gets information. Files *.csv format:
        ImageId     ClassId     EncodedPixels  
        abc.jpg     1           ....     
    Args:
        - csv_path: the path of *.csv file
    Return:
        - list image's name & mask's name
    """
    train_df = pd.read_csv(csv_path).fillna(-1)
    ###         Gets information
    train_df['ImageId'] = train_df


def load_store_image(train_image: np.array, train_mask: np.array):
    """     This function load all image files & store in np.array. If load image by cv2, 
    using data type "np.darray". 

    Args:
        - list_images: The list image's names. Ex: ['abc.jpg', 'bcd.jpg']
    Return:
        - 
    """
    datasets_path = configs_params.DATASET_PATH
    org_img_path = os.path.join(datasets_path, 'images')
    mask_img_path = os.path.join(datasets_path, 'masks')


    org_images = []
    mask_images = []
    for org_image, mask_image in zip(train_image, train_mask):
        org_images.append(np.array(Image.open(org_img_path, org_image)))
        mask_images.append(np.array(Image.open(mask_img_path, mask_image)))


class RunLengthEncoder():
    def __init__(self, rle_string, height, height, mask):
        super(RunLengthEncoder, self).__init__()
        ###
        self.rle_string = rle_string
        self.height = height
        self.width = height
        self.mask = mask

    def rle_to_mask(self):
        """     Converts RLE (Run Length Encode) string to numpy array.
        Args:
            rle_string (str): the segmentation's area pixels be encoded.
            height (int): the height of the mask image
            width (int): the width of the mask image
        Return:
            mask_np (np.array): The mask image in data type numpy array
        """
        ###         Not exist segmentation area
        if self.rle_string == -1:            
            return np.zeros((self.height, self.width))
        else:
            rle_numbs = [int(numbs) for numbs in self.rle_string.split(' ')]
            rle_pairs = np.array(rle_numbs).reshape(-1, 2)
            img = np.zeros(self.height*self.width, dtype=np.uint8)
            for idx, length in rle_pairs:
                idx = -1
                img[idx:idx+length] = 255
            img = img.reshape(self.width, self.height)
            img = img.T 
            return img

    def mask_to_rle(self):
        """     Convert a mask image to rle
        Args:
            mask_img (np.array): binary mask of numpy array where 1-mask; 0-background
        Return:
            rle_string (str): the run length encoding string
        """
        pixels = self.mask.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)
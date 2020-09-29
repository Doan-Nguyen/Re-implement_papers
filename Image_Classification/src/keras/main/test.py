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
##              Logging


def test_single_image(img_path):
    """         Test predict a image
    Args:
        img_path (str): the image's path
    Return:
        class_score (float): the predict's probability
    """
    test_img = tf.io.read_file(img_path)
    img_tensor = tf.image.decode_jpeg(contents=test_img, channels=configs_param.channels)
    img_tensor = tf.image.resize(img_tensor, [configs_param.image_height, configs_param.image_width])

    img_numpy = img_tensor.numpy()
    img_numpy = (np.expand_dims(img_numpy, 0))
    img_tensor = tf.convert_to_tensor(img_numpy, tf.float32)
    img = img_tensor/255.0
    prob = model(img)

    cls_prob = np.argmax(prob)

    return cls_prob

if __name__ == '__main__':
    model = tf.keras.models.load_model(os.path.join(configs_param.CKPT_DIR, configs_param.MODEL_RESULTS))
    cls = test_single_image('/home/doannn/Documents/Public/bitbucket/datasets/cifar100_imgs/test_image/0005.png')
    print(cls)
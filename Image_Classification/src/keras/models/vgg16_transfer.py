##              Basic libaries
import numpy as np
import os
import sys
from pathlib import Path
##              Framework
import tensorflow.keras as keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Flatten, Dense
##              Files
current_path = Path(os.getcwd())
sys.path.append(str(current_path))
import configs_param
##              Logging


def vgg_pretrained():
    ###     Get pre-trained VGG-16
    model_vgg16_conv = VGG16(
            weights='imagenet',
            include_top=False)  # remote the last fully connected layer
    model_vgg16_conv.summary()
    #   get new inputs datasets
    input = Input(
            shape=(configs_param.image_width, configs_param.image_height, configs_param.channels),
            name='image_input')
    #   use the generated model
    output_vgg16_conv = model_vgg16_conv(input)
    
    #   Add fully-connected layers with random initializer
    x = keras.layers.Flatten(data_format=None)(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(configs_param.NUM_CLASSES, activation='softmax', name='predictions')(x)

    
    new_model = Model(input=input, output=x)
    new_model.summary()

    return new_model

if __name__ == "__main__":
        vgg_pretrained()
    
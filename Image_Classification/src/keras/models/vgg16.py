##              Basic libaries
import os
import sys
##              Framework
import tensorflow as tf
import tensorflow.keras.layers as layers
##              Files
from pathlib import Path
current_path = Path(os.getcwd())
sys.path.append(str(current_path))
import configs_param


def VGG16_scratch():
    model = tf.keras.Sequential([
        ###             Feature maps
        tf.keras.layers.Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=1,
                            padding='same',
                            activation=tf.keras.activations.relu,
                            input_shape=(configs_param.image_height, configs_param.image_width, configs_param.channels)),
        tf.keras.layers.Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=1,
                            padding='same',
                            activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                            strides=2,
                            padding='same'),
        # 2 
        tf.keras.layers.Conv2D(filters=128,
                            kernel_size=(3, 3),
                            strides=1,
                            padding='same',
                            activation=tf.keras.activations.relu),
        tf.keras.layers.Conv2D(filters=128,
                            kernel_size=(3, 3),
                            strides=1,
                            padding='same',
                            activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                            strides=2,
                            padding='same'),
        # 3
        tf.keras.layers.Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=1,
                            padding='same',
                            activation=tf.keras.activations.relu),
        tf.keras.layers.Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=1,
                            padding='same',
                            activation=tf.keras.activations.relu),
        tf.keras.layers.Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=1,
                            padding='same',
                            activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                            strides=2,
                            padding='same'),
        # 4
        tf.keras.layers.Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=1,
                            padding='same',
                            activation=tf.keras.activations.relu),
        tf.keras.layers.Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=1,
                            padding='same',
                            activation=tf.keras.activations.relu),
        tf.keras.layers.Conv2D(
                filters=512,
                kernel_size=(3, 3),
                strides=1,
                padding='same',
                activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(
                pool_size=(2, 2),
                strides=2,
                padding='same'),
        # 5
        tf.keras.layers.Conv2D(
                filters=512,
                kernel_size=(3, 3),
                strides=1,
                padding='same',
                activation=tf.keras.activations.relu),
        tf.keras.layers.Conv2D(
                filters=512,
                kernel_size=(3, 3),
                strides=1,
                padding='same',
                activation=tf.keras.activations.relu),
        tf.keras.layers.Conv2D(
                filters=512,
                kernel_size=(3, 3),
                strides=1,
                padding='same',
                activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(
                pool_size=(2, 2),
                strides=2,
                padding='same'),
        
        ###         Classifiers
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(
                units=4096,
                activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(
                units=4096,
                activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(
                units=configs_param.NUM_CLASSES,
                activation=tf.keras.activations.softmax)
    ])

    return model

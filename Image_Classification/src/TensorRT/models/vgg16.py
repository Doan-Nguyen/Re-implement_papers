#   basic packages
import sys
#   import framework
import tensorflow as tf 
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
#   import files
sys.path.append('../config_model')
# from config_model import NUM_CLASSES, image_width, image_height, channels
import config_model 


def VGG16():
    """             VGG-16
    +) Activation function: ReLU()
    """
    model = tf.keras.Sequential([
    ###         Config architecture
        layers.Conv2D(
                filters=64,         # 
                kernel_size=(3, 3),
                activation=activations.relu,
                input_shape=config_model.input_shape),
        layers.Conv2D(
                filters=64, 
                kernel_size=(3, 3),
                activation=activations.relu),
        layers.MaxPool2D(
                pool_size=(2, 2))

        layers.Conv2D(
                filters=128,         # 
                kernel_size=(3, 3),
                activation=activations.relu),
        layers.Conv2D(
                filters=128, 
                kernel_size=(3, 3),
                activation=activations.relu),
        layers.MaxPool2D(
                pool_size=(2, 2))
    ])
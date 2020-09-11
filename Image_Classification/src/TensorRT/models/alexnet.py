#   basic packages
import sys
#   import framework
import tensorflow as tf 
import tensorflow.keras.layers as layers
#   import files
sys.path.append('../config_model')
from config_model import NUM_CLASSES, image_width, image_height, channels


def AlexNet():
    """         The architecture 
    Contains 8 layers {5 conv + 3 fc}
    + Conv1:
        - conv(in_ch=227, out_ch=55, kernel=11, stride=4, padding='valid) # valid ~ padding=0
        - MaxPooling(kernel=3, stride=2)
        - BatchNorm() -> out_ch = 27
    + Conv2: 
        - conv(in_ch=27, out_ch=27, kernel=5, stride=4)
        - MaxPooling(kernel=3, stride=2)
        - BatchNorm()
    """
    model = tf.keras.Sequential([
        ###             First convolution layer
        layers.Conv2D(
                    filters=96, 
                    kernel_size=11,
                    strides=4, 
                    padding='valid', 
                    activation=tf.keras.activations.relu,
                    input_shape=(image_height, image_width, channels)
        ),
        layers.MaxPooling2D(
                    pool_size=3,
                    strides=2,
                    padding='valid'
        ),
        layers.BatchNormalization(),
        ###             Second convolution layer
        layers.Conv2D(
                    filters=256,
                    kernel_size=5,
                    strides=1,
                    padding='same', # padding='same'
                    activation=tf.keras.activations.relu
        ),
        layers.MaxPooling2D(
                    pool_size=3,
                    strides=2,
                    padding='same'
        ),
        layers.BatchNormalization(),
        ###             Third convolution layer
        layers.Conv2D(
                    filters=384,
                    kernel_size=3, 
                    strides=1,
                    padding='same',
                    activation=tf.keras.activations.relu
        ),
        ###         4th convolution layer
        layers.Conv2D(
                    filters=384,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    activation=tf.keras.activations.relu
        ),
        ###         Fith convolution layer
        layers.Conv2D(
                    filters=256, 
                    kernel_size=3, 
                    padding='same', 
                    strides=1,
                    activation=tf.keras.activations.relu
        ),
        layers.MaxPooling2D(
                    pool_size=3,
                    strides=2,
                    padding='same'
        ),
        layers.BatchNormalization(),
        ###         6th fully connected layer
        layers.Flatten(),
        layers.Dense(units=4096,
                    activation=tf.keras.activations.relu
        ),
        layers.Dropout(rate=0.2),
        ###         7th fully connected layer
        layers.Flatten(),
        layers.Dense(units=4096,
                    activation=tf.keras.activations.relu
        ),
        layers.Dropout(rate=0.2),
        ###         8th fully connected layer
        layers.Dense(units=NUM_CLASSES, 
                    activation=tf.keras.activations.softmax)
    ])

    return model
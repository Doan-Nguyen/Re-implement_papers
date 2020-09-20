#       Basic packages
import sys
#       Framework
import tensorflow as tf 
import tensorflow.keras as keras
#       Files
import vgg 


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    """
    This function to normalization training datasets. 
    Makesure datasets same size.
    """
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError("Len(means) must match the number of channels")
    ##      splits a tensor value into a list sub-tensors
    channels = tf.split(
        axis=3, 
        num_or_size_splits=num_channels, 
        value=images
    )
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)




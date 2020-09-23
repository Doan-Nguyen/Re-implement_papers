# -*- coding: utf-8 -*-
"""        Contains a set of utilities that allow building the UNet model           """
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers 


def _crop_concat(inputs: tf.Tensor, residual_input: tf.Tensor) -> tf.Tensor:
    """

    """
    factor = inputs.shape[1] / residual_input.shape[1]
    return tf.concat([inputs, tf.image.central_crop(residual_input, factor)], axis=-1)


class InputBlock(keras.Model):
    """         Encoder Path

    """
    def __init__(self, filters):
        super(InputBlock, self).__init__()
        ###
        with tf.name_scope("input_block"):
            self.conv1 = layers.Conv2D(
                    filters=filters,
                    kernel_size=(3, 3),
                    activation=tf.nn.relu
            )
            self.conv2 = layers.Conv2D(
                    filters=filters,
                    kernel_size=(3, 3),
                    activation=tf.nn.relu
            )
            self.maxpool = layers.MaxPooling2D(
                    pool_size=(2, 2), 
                    strides=2)
    
    def call(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output) 
        mp = self.maxpool(output)
        return mp, output


class DownsampleBlock(keras.Model):
    def __init__(self, filters, idx):
        super(DownsampleBlock, self).__init__()
        ###
        with tf.name_scope('Downsample_block_{}'.format(idx)):
            self.conv1 = layers.Conv2D(
                    filters=filters,
                    kernel_size=(3, 3),
                    activation=tf.nn.relu
            )
            self.conv2 = layers.Conv2D(
                    filters=filters,
                    kernel_size=(3, 3),
                    activation=tf.nn.relu
            )
            self.maxpool = layers.MaxPooling2D(
                    pool_size=(2, 2), 
                    strides=2)

    def call(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output) 
        mp = self.maxpool(output)
        return mp, output


class BottleneckBlock(keras.Model):
    def __init__(self, filters, idx):
        super(DownsampleBlock, self).__init__()
        ###
        with tf.name_scope('Downsample_block_{}'.format(idx)):
            self.conv1 = layers.Conv2D(
                    filters=filters,
                    kernel_size=(3, 3),
                    activation=tf.nn.relu
            )
            self.conv2 = layers.Conv2D(
                    filters=filters,
                    kernel_size=(3, 3),
                    activation=tf.nn.relu
            )
            # self.dropout = layers.Dropout(rate=0.2)
            self.conv_transpose = layers.Conv2DTranspose(
                    filters=filters //2, 
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding='same', 
                    activation=tf.nn.relu
            )

    def call(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output) 
        output = self.conv_transpose(output)
        return output


class UpsampleBlock(keras.Model):
    def __init__(self, filters, idx):
        super(UpsampleBlock, self).__init__()
        ###
        with tf.name_scope('Upsample_block_{}'.format(idx)):
            self.conv1 = layers.Conv2D(
                    filters=filters,
                    kernel_size=(3, 3),
                    activation=tf.nn.relu
            )
            self.conv2 = layers.Conv2D(
                    filters=filters,
                    kernel_size=(3, 3),
                    activation=tf.nn.relu
            )
            # self.dropout = layers.Dropout(rate=0.2)
            self.conv_transpose = layers.Conv2DTranspose(
                    filters=filters //2, 
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding='same', 
                    activation=tf.nn.relu
            )

    def call(self, inputs, residual_input):
        output = _crop_concat(inputs, residual_input)
        output = self.conv1(inputs)
        output = self.conv2(output) 
        output = self.conv_transpose(output)
        return output



class OutputBlock(tf.keras.Model):
    def __init__(self, filters, n_classes):
        super().__init__(self)
        with tf.name_scope('output_block'):
            self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=(3, 3),
                                                activation=tf.nn.relu)
            self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=(3, 3),
                                                activation=tf.nn.relu)
            self.conv3 = tf.keras.layers.Conv2D(filters=n_classes,
                                                kernel_size=(1, 1),
                                                activation=None)

    def call(self, inputs, residual_input):
        out = _crop_and_concat(inputs, residual_input)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

import cv2
import tensorflow as tf 
import vgg 
import resnet
import config 
import pdb

class PixelLinkNet(object):
    def __init__(self, inputs, weight_decay= None, basenet= "resnet", 
                data_format= 'NHWC', 
                weight_initializer, 
                biases_initializer):
        """

        """
        self.inputs = inputs
        self.weight_decay = weight_decay
        self.basenet_type = basenet, 
        self.data_format = data_format

        """          Choose weights & biasese initializer
        in tf2: GlorotUniform() ~ xavier_initializer
        """
        self.weight_initializer = tf.initializers.GlorotUniform()
        self.biases_initializer = tf.zeros_initializer()

        self._build_network()
        self.shapes = self.get_shapes()
    
    def get_shapes(self):
        shapes = {}

        for layer in self.end_points:
            shapes[layer] = tensor_shape(self.end_points[layer])[1:-1]
        return shapes
    
    def get_shape(self, name):
        return self.shapes[name]
    
    def unpool(self, inputs):
        return tf.compat.v1.image.resize_bilinear(
            inputs, 
            size= [tf.shape(inputs)[1]*2, tf.shape(inputs)[2]*2])

    def _build_network(self):
        
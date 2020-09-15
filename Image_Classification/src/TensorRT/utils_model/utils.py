import tensorflow as tf 


def network_initializer(self):
    """

    """
    with tf.variable_creator_scope('convolution_network') as scope:
        output = self.
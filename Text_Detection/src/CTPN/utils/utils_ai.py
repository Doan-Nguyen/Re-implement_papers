import tensorflow as tf 


def make_var(name, shape, initializer=None):
    """ A variable maintains shared. Variable() constructor requires
     an initial value for the variable, defines the type & shape of the variable.

    """
    return tf.compat.v1.get_variable(name=name, shape=shape, initializer=initializer)

def anchor_target_layer(cls_pre, bbox, img_info, scope_name):
    """     Get the results of anchor
    Args:
        cls_pre (float): classifier predict
        bbox : bounding boxes
        img_info: image information
    """

    with tf.variable_creator_scope(scope_name) as scope:
        

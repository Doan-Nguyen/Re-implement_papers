import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.framework import graph_io

def keras_to_frozen_pb(model_path, 
                        model_out_path,
                        custom_object_dict=None,
                        tensor_out_name=None,
                        tensorboard_dir=None):
    """         Converter keras model to frozen pb model.
    Args:
        model_path: input model path (*.h5)
        model_output_path: output model path (*.pb)
        tensor_out_name (str, optional): Specified name of output tensor
        tensor_board_dir (str, optional): Output tensorboard dir path
    """
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session()
        K.set_session()
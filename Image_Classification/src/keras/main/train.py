##              Basic libaries
import sys
import os
from pathlib import Path
from datetime import datetime
from packaging import version
import numpy as np
##              Framework
import tensorflow as tf
import tensorflow.keras as keras
tf.config.experimental.list_physical_devices('GPU')
##              Files
current_path = Path(os.getcwd())
sys.path.append(str(current_path))
import configs_param
from utils_ai.process_folders import check_make_folder
from models.simple_convnet import simple_model
from utils_datasets.dataset_preprocess import datasets_generator
##              Logging

def tensorboard():
    logdir = check_make_folder(configs_param.LOG_DIR)
    log_timing = os.path.join(logdir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_timing)
    
    return tensorboard_callback


def train():
    ###         Load datasets
    train_generator, test_generator = datasets_generator()
    ###         Load model
    model = simple_model()
    ###         Logging to tensorboard
    tensorboard_callback = tensorboard()
    ###
    model.fit_generator(
        train_generator, 
        steps_per_epoch=100// configs_param.BATCH_SIZE,
        epochs= 50,
        validation_data=test_generator,
        validation_steps=70 // configs_param.BATCH_SIZE,
        callbacks=[tensorboard_callback]
    )
    model.save_weights(os.path.join(configs_param.CKPT_DIR, configs_param.MODEL_RESULTS))

if __name__ == "__main__":
    train()
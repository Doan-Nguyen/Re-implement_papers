#   basic packages
from __future__ import absolute_import, division, print_function
import sys 
import os 
import datetime
#   framework
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import TensorBoard
#   import files
from config_model import *
from datasets.prepare_data import get_datasets
from models.alexnet import AlexNet



def get_model():
    # model = VGG16()
    model = AlexNet()

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

def train(model): 
    """             Using Tensorboard
    Use command: $ tensorboard: --logdir "logs"
    """   
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log')
    callback_list = [tensorboard]

    ###             Training
    model.fit_generator(train_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=train_num // BATCH_SIZE,
                        validation_data=valid_generator,
                        validation_steps=valid_num // BATCH_SIZE,
                        callbacks=callback_list)

    ###             Save the whole model
    model.save(model_dir)


if __name__ == "__main__":
    ###         GPU settings
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    ###         Datasets
    # train_generator, valid_generator, test_generator, train_num, valid_num, test_num = get_datasets()
    train_generator, valid_generator, train_num, valid_num = get_datasets()


    ###         Get model
    model = get_model()
    model.summary()
    ###
    train(model)
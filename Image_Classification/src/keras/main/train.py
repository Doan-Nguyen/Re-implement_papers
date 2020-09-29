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
##              Files
current_path = Path(os.getcwd())
sys.path.append(str(current_path))
import configs_param
from utils_ai.process_folders import check_make_folder
from models.vgg16 import VGG16_scratch
from utils_datasets.dataset_preprocess import get_datasets
##              Logging


def tensorboard():
    logdir = check_make_folder(configs_param.LOG_DIR)
    log_timing = os.path.join(logdir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_timing)
    return tensorboard_callback


def ckpt_callback():
    cp_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(configs_param.CKPT_DIR, configs_param.MODEL_RESULTS),
            save_weights_only=True,
            verbose=1)
    return cp_callback


def get_model():
    # model = AlexNet()
    model = VGG16_scratch()
    # pretrained_model = vgg16_pretrained.vgg_pretrained()
    # model = VGG19()
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    ##      GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print("Using GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train_generator, valid_generator, train_num, valid_num = get_datasets()

    # Use command tensorboard --logdir "log" to start tensorboard
    # cp_callback = ckpt_callback()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='log')
    callbacks_list = [tensorboard_callback]

    checkpoint_path = os.path.join(configs_param.CKPT_DIR, configs_param.MODEL_RESULTS)
    model = get_model()
    # start training
    model.fit_generator(train_generator,
            epochs=configs_param.EPOCHS,
            steps_per_epoch=train_num // configs_param.BATCH_SIZE,
            validation_data=valid_generator,
            validation_steps=valid_num // configs_param.BATCH_SIZE,
            callbacks=callbacks_list)

    # save the whole model
    model.save('./checkpoint.h5')
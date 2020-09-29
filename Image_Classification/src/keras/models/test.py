##              Basic libaries
from __future__ import absolute_import, division, print_function
import os
import sys
##              Framework
import tensorflow as tf
##              Files
from vgg16 import VGG16

from pathlib import Path
current_path = Path(os.getcwd())
sys.path.append(str(current_path))
from configs_param import BATCH_SIZE, EPOCHS, CKPT_DIR
# from config import EPOCHS, BATCH_SIZE, model_dir
from utils_datasets.dataset_preprocess import get_datasets
# from models.alexnet import AlexNet
# from models.vgg19 import VGG19

def get_model():
    # model = AlexNet()
    model = VGG16()
    # model = VGG19()

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train_generator, test_generator, \
    train_num, test_num = get_datasets()

    # Use command tensorboard --logdir "log" to start tensorboard
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log')
    callback_list = [tensorboard]

    model = get_model()
    model.summary()

    # start training
    model.fit_generator(train_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=train_num // BATCH_SIZE,
                        validation_data=test_generator,
                        validation_steps=test_num // BATCH_SIZE,
                        callbacks=callback_list)

    # save the whole model
    model.save(CKPT_DIR)

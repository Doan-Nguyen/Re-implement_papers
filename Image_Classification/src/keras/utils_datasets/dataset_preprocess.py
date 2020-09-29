##              Basic libaries
import sys
import os
##              Framework
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
##              Files
from pathlib import Path
current_path = Path(os.getcwd())
sys.path.append(str(current_path))
import configs_param


def get_datasets():
    # Preprocess the dataset
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0/255.0)
    train_generator = train_datagen.flow_from_directory(
                configs_param.TRAIN_DIR,
                target_size=(configs_param.image_height, configs_param.image_width),
                color_mode="rgb",
                batch_size=configs_param.BATCH_SIZE,
                seed=7,
                shuffle=True,
                class_mode="categorical")
        

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0/255.0)
    valid_generator = valid_datagen.flow_from_directory(
                configs_param.VALID_DIR,
                target_size=(configs_param.image_height, configs_param.image_width),
                color_mode="rgb",
                batch_size=configs_param.BATCH_SIZE,
                seed=7,
                shuffle=True,
                class_mode="categorical")

    train_num = train_generator.samples
    valid_num = valid_generator.samples

    return train_generator, valid_generator, train_num, valid_num




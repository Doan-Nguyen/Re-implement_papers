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


def datasets_generator():
    """     This function generator datasets
    Return:
        train_generator, test_generator
    """
    ##          the augmentation configuration 
    train_datagen = ImageDataGenerator(
                # rotation_range=40,        
                # width_shift_range=1,
                # height_shift_range=1,
                rescale=1/255,
                # zoom_range=1,
                # horizontal_flip=False,
                fill_mode='nearest')
    
    test_datagen = ImageDataGenerator(
                # rotation_range=40,        
                # width_shift_range=1,
                # height_shift_range=1,
                rescale=1/255,
                # zoom_range=1,
                # horizontal_flip=False,
                fill_mode='nearest')

    ##          Read images in datasets folder
    train_generator = train_datagen.flow_from_directory(
            directory=configs_param.TRAIN_DIR, 
            batch_size=configs_param.BATCH_SIZE,
            class_mode='binary')

    test_generator = train_datagen.flow_from_directory(
            directory=configs_param.TEST_DIR, 
            batch_size=configs_param.BATCH_SIZE,
            class_mode='binary')

    return train_generator, test_generator


def get_datasets():
    # Preprocess the dataset
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
                configs_param.TRAIN_DIR,
                target_size=(configs_param.image_height, configs_param.image_width),
                color_mode="rgb",
                batch_size=configs_param.BATCH_SIZE,
                seed=1,
                shuffle=True,
                class_mode="categorical")

#     valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#         rescale=1.0 /255.0)

#     valid_generator = valid_datagen.flow_from_directory(
#                 configs_param.valid_dir,
#                 # target_size=(configs_param.image_height, configs_param.image_width),
#                 color_mode="rgb",
#                 batch_size=configs_param.BATCH_SIZE,
#                 seed=7,
#                 shuffle=True,
#                 class_mode="categorical")
        
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 /255.0)

    test_generator = test_datagen.flow_from_directory(
                configs_param.TEST_DIR,
                target_size=(configs_param.image_height, configs_param.image_width),
                color_mode="rgb",
                batch_size=configs_param.BATCH_SIZE,
                seed=7,
                shuffle=True,
                class_mode="categorical"
                )


    train_num = train_generator.samples
#     valid_num = valid_generator.samples
    test_num = test_generator.samples


    return train_generator, \
        test_generator, \
        train_num, test_num

# return train_generator, \
# valid_generator, \
# test_generator, \
# train_num, valid_num, test_num



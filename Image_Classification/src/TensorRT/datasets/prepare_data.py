#   basic packages
import sys 
import os 
#   framework
import tensorflow.keras as keras
#   import files
sys.path.append('../')
import config_model

def get_datasets():
    """            This function gets datasets for model
    Preprocess the datasets.
    """
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0/255.0)
    train_generator = train_datagen.flow_from_directory(
            config_model.train_dir, 
            target_size=(config_model.image_height, config_model.image_width),
            color_mode='rgb',
            batch_size=config_model.BATCH_SIZE, 
            seed=1,
            shuffle=True,
            class_mode="categorical"
    )
    ###
    valid_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0/255.0)
    valid_generator = valid_datagen.flow_from_directory(
            config_model.valid_dir, 
            target_size=(config_model.image_height, config_model.image_width),
            color_mode='rgb',
            batch_size=config_model.BATCH_SIZE, 
            seed=7,
            shuffle=True,
            class_mode="categorical"
    )
    ###
#     test_datagen = keras.preprocessing.image.ImageDataGenerator(
#             rescale=1.0/255.0)
#     test_generator = test_datagen.flow_from_directory(
#             config_model.test_dir, 
#             target_size=(config_model.image_height, config_model.image_width),
#             color_mode='rgb',
#             batch_size=config_model.BATCH_SIZE, 
#             seed=7,
#             shuffle=True,
#             class_mode="categorical"
#     )

    train_num = train_generator.samples
    valid_num = valid_generator.samples
#     test_num = test_generator.samples

    return train_generator, \
           valid_generator, \
           train_num, valid_num



#     return train_generator, \
#            valid_generator, \
#            test_generator, \
#            train_num, valid_num, test_num
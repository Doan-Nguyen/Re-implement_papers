from model import *
from data import *

### 
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
                    

myGene = trainGenerator(1, 
                        '/media/doannn/Data1/Work/Projects/Teeth/Datasets/data_dataset_voc_bose/bone',
                        'imgs_results','masks_results',
                        data_gen_args,
                        save_to_dir = None)

def train():
    model = unet()
    model_checkpoint = ModelCheckpoint('bone_489_1920.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=600, epochs=100, callbacks=[model_checkpoint])


if __name__ == "__main__":
    train()


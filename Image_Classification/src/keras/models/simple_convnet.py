##              Basic libaries
import sys
##              Framework
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
##              Files
from pathlib import Path
current_path = Path(os.getcwd())
sys.path.append(str(current_path))
import configs_param


def simple_model():
    model = Sequential()
    ###         Features map 3D (height, width, features) -> model outputs 3D
    model.add(Conv2D(filters=32, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    ###         Classification: converts features map 3D -> features vector 1D
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    ###         Compile the model
    model.compile(
        loss='binary_crossentropy', 
        optimizer='rmsprop', 
        metrics=['accuracy']
    )

    return model


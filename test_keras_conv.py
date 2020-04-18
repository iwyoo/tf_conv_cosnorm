import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(8)

import keras
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from keras import regularizers
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras import optimizers

from sim_layer import Norm_Conv2d as Conv2D

# create custom activation function for CosSim layer

def pow_activation(x, p_val=3):
    #return 2 * K.pow(K.tanh(x) , p_val)
    return K.pow(x, p_val)
    
get_custom_objects().update({'pow_activation': Activation(pow_activation)})


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 25:
        lrate = 0.0005
    if epoch > 50:
        lrate = 0.0002
    return lrate


batch_size = 32
epochs = 75

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = (x_train / 255).astype('float32')
x_test = (x_test / 255).astype('float32')

model = Sequential()
model.add(Conv2D(128, (5, 5), padding='same',
                 input_shape=x_train.shape[1:], kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation(pow_activation))
model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation(pow_activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation(pow_activation))
model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation(pow_activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(10))
model.add(Activation('softmax'))

adam = optimizers.Adam(lr=1e-3)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )

datagen.fit(x_train)
    
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs,validation_data=(x_test, y_test),
                    workers=4, callbacks=[LearningRateScheduler(lr_schedule)])

model.save_weights("test_w_gpu.h5")

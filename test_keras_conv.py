import os
from sim_layer import Norm_Conv2d as Conv2D

import keras
import keras.backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras import optimizers



# create custom activation function for CosSim layer
from keras.utils.generic_utils import get_custom_objects

def pow_activation(x, p_val=5):
    return K.pow(x, p_val)

get_custom_objects().update({'pow_activation': Activation(pow_activation)})

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = (x_train / 255).astype('float32')
x_test = (x_test / 255).astype('float32')

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation(pow_activation))
model.add(Conv2D(32, (3, 3)))
model.add(Activation(pow_activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation(pow_activation))
model.add(Conv2D(64, (3, 3)))
model.add(Activation(pow_activation))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(10))
model.add(Activation('softmax'))



# compile and test model
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_test, y_test),
          shuffle=True)

print(model.evaluate(x_test, y_test))

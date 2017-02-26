#!/usr/bin/python
# -*- coding: utf-8 -*
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import sys

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.load_weights('first_try.h5')  # always save your weights after training or during training

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=100, 
        classes=['dogs'],
        class_mode="binary")

##img_path = 'data/validation/dogs/dog.9648.jpg'
##img = image.load_img(img_path, target_size=(150, 150))
##x = image.img_to_array(img)
##x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)

preds = model.predict_generator(test_generator, 100)
#preds = predict_on_batch(pg)
##preds = model.predict(x)

nn = preds.flatten()
# print('preds:', nn)

print('Predicted:', np.mean(nn))

K.clear_session()

#ef preprocess_input(x):
#    return x.reshape((-1, ) + input_shape) / 255.
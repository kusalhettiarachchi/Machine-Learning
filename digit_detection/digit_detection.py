#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:40:04 2018

@author: rabbie
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils


batch_size=128
nb_classes=10 #number of classes the NN has to classify the images to ( 0 to 9 )

#input image dimensions
img_rows,img_cols = 28,28

#splitting data 
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#reshaping data
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#covert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('One hot encoding: {}'.format(Y_train[0,:]))

model = Sequential()

#adding convolutional layers
model.add(Convolution2D(6, 5, 5, input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(16, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(120, 5, 5))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))  

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

nb_epoch = 2 #number of iterations

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score: ', score[0])
print('Test accuracy: ', score[1])

#visualizing some samples
def show_samples():
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(X_train[i,0], cmap='gray')
        plt.axis('off')

#visualizing some results
def show_results():
    res = model.predict_classes(X_test[:9])
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(X_train[i,0], cmap='gray')
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.ylabel('Prediction: {}'.format(res[i]))
























































































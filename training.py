#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 21:02:54 2017

@author: saurabh
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import Model 
from keras.models import load_model
from keras.applications.vgg16 import VGG16

def preProcess():
    
    img_width, img_height = 300, 300
    
    train_data_dir = 'data/train'
    validation_data_dir = 'data/val'
    nb_train_samples = 450
    nb_validation_samples = 150
    batch_size = 1
    
    
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1 / 255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')#*************
    
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')#**************
    
    
    
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    
    return input_shape, train_generator, validation_generator, nb_train_samples, nb_validation_samples
    

def buildModel(input_shape, train_generator, validation_generator, nb_train_samples, nb_validation_samples):
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3))#*************
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',#*************
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    epochs = 50
    batch_size = 1
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        verbose= 1)
    
    return model

def getModel(input_shape, train_generator, validation_generator, nb_train_samples, nb_validation_samples):
    try:
        model = load_model('hand.h5')
        model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    except:
        model = buildModel(input_shape, train_generator, validation_generator, nb_train_samples, nb_validation_samples)
        
    return model


if __name__ == '__main__':
    '''
    input_shape, train_generator, validation_generator, nb_train_samples, nb_validation_samples = preProcess()
    model = getModel(input_shape, train_generator, validation_generator, nb_train_samples, nb_validation_samples)
    '''
    print('training loaded!!!')
    
    
    
'''
model = VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))


for layer in model.layers:
    layer.trainable = False

x = model.output
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.2)(x)

predictions = Dense(3, activation="softmax")(x)
model_final = Model(input = model.input, output = predictions)

history = model_final.compile(loss = "categorical_crossentropy", optimizer = 'rmsprop', metrics=["accuracy"])

'''
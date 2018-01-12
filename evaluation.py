#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 21:59:46 2017

@author: saurabh
"""
import training
import cv2
import numpy as np
import skinMask, binaryMask
from keras.models import load_model

i = 0
def getResult(img, model):
    
    label = model.predict(img)
    #print(label)
    
    if int(label[0][0]) :
        return 'closed'
    
    if int(label[0][1]) :
        return 'one'
    
    return 'palm'
    

if __name__ == '__main__':
    
    x0 = 100
    y0 = 100
    height = 300
    width = 300
    
    cap = cv2.VideoCapture(0)
    skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    try:
        model = load_model('hand.h5')
        model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    except:
        input_shape, train_generator, validation_generator, nb_train_samples, nb_validation_samples = training.preProcess()
        model = training.getModel(input_shape, train_generator, validation_generator, nb_train_samples, nb_validation_samples)
    
#    model.pop()
    
    model.compile(loss='categorical_crossentropy',
                 optimizer='rmsprop',
                 metrics=['accuracy'])
    while(True):
        ret, frame = cap.read()
        if ret == True:
            
            roi = skinMask.skinMask(frame, x0, y0, width, height, skinkernel)
            
            
            final = binaryMask.binaryMask(roi)
            cv2.imshow('skinMask', roi)
            cv2.imshow('bmask',final)
            
            
            cv2.imwrite('data/test/test'+str(i)+'.jpg', final)
            final = cv2.imread('data/test/test'+str(i)+'.jpg')
            final = final.reshape((1,300,300,3))
            label = getResult(final, model)
            i += 1
            print(label)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            
        else:
            print("Ret is false")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
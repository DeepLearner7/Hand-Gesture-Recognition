##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Thu Dec 21 01:13:19 2017
#
#@author: saurabh
#
#============Info for ubuntu=================
#pip install opencv-python
#see version by pip freeze.
#Otherwise install by-> conda install -c anaconda opencv , for version 3.3.1
#Read for more info - > https://github.com/ContinuumIO/anaconda-issues/issues/121
#gtk errror: https://stackoverflow.com/questions/14655969/opencv-error-the-function-is-not-implemented
#            https://stackoverflow.com/questions/28776053/opencv-gtk2-x-error
#
#Perfect solution https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/
#"""
#
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import binaryMask, skinMask
cap = cv2.VideoCapture(0)


x0 = 100
y0 = 100
height = 300
width = 300

skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

i = 1

while(i):
    ret, frame = cap.read()
    if ret == True:
        
        roi = skinMask.skinMask(frame, x0, y0, width, height, skinkernel)
        cv2.imshow('skin',roi)
        
        final = binaryMask.binaryMask(roi)
        cv2.imshow('bmask',final)
        
        print(i)
        if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        i += 1
        time.sleep(0.2)
        cv2.imwrite('extra/closed'+str(i)+'.jpg', final)
    else:
        
        print("Ret is false")
        break
    

cap.release()
cv2.destroyAllWindows()
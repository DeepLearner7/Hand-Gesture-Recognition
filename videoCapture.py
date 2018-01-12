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

#cap = cv2.VideoCapture(0)
#
#i=1
#fgbg = cv2.createBackgroundSubtractorMOG2()
#
#if (cap.isOpened()== False): 
#  print("Error opening video stream")
# 
#
#while(i):
#    ret, frame = cap.read()
#    if ret == True:
##        cv2.imshow('Frame:', frame)
##        edges = cv2.Canny(frame, 100, 150)
#        #cv2.imshow('Edges:', edges)
##        grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
##        cv2.imshow('grayScale', grayScale)
##        _, thresh = cv2.threshold(grayScale, 100, 255, cv2.THRESH_BINARY)
###        cv2.imshow('thresh', thresh)
##        _, threshGray = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY)
##        _, thresh = cv2.threshold(threshGray, 122, 255, cv2.THRESH_BINARY)
##        
#        roi = frame[y0:y0+height, x0:x0+width]
#        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
##        cv2.imshow('threshGray', threshGray)
#        
#        
##        gaus = cv2.adaptiveThreshold(grayScale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
##        cv2.imshow('gaus',gaus)
##        fgmask = fgbg.apply(threshGray)
##        fgmask_original = fgbg.apply(frame)
##        cv2.imshow('fgbg', gaus)
##        cv2.imshow('fgmask', fgmask)
##        cv2.imshow('fgmask_original', fgmask_original)
##        cv2.imshow('sobely', hsv)
##        
#
#        skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
#        low_range = np.array([0, 50, 80])
#        upper_range = np.array([30, 200, 255])
##        
#        mask1 = cv2.inRange(hsv, low_range, upper_range)
#        mask2 = cv2.erode(mask1, skinkernel, iterations = 1)
#        mask3 = cv2.dilate(mask2, skinkernel, iterations = 1)
##        
#        mask4 = cv2.GaussianBlur(mask3, (5,5), 1)
##        res = cv2.bitwise_and(roi, roi, mask4)
#
#        
##        blur = cv2.GaussianBlur(hsv, (5,5), 1)
##        fgmask = fgbg.apply(mask4)
##        _,th3 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
##        
##        mask = cv2.inRange(hsv, lower_red, upper_red)
##        res = cv2.bitwise_and(frame,frame, mask= mask)
##        
##        kernel = np.ones((5,5),np.uint8)
##        erosion = cv2.erode(mask,kernel,iterations = 1)
##        dilation = cv2.dilate(mask,kernel,iterations = 1)
###
##        cv2.imshow('Original',mask)
##        cv2.imshow('Mask',res)
##        cv2.imshow('Erosion',erosion)
##        cv2.imshow('Dilation',dilation)
##            
#        #blur = cv2.GaussianBlur(grayScale,(5,5),0)
##        cv2.imshow('Blured',blur)
##        ret,thresh1 = cv2.threshold(grayScale,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#        cv2.imshow('fgmask',mask3)
##        cv2.imshow('mask4',mask4)
##        cv2.imwrite('two/one'+str(i)+'.jpg', mask4)
#        i += 1
##        time.sleep(0.2)
#        
#        
#        if cv2.waitKey(25) & 0xFF == ord('q'):
#            break
#        
#    else:
#        print("Ret is false")
#        break
#    
#cap.release()
#cv2.destroyAllWindows()

def skinMask(frame, x0, y0, width, height, skinkernel):
    
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    
    mask = cv2.inRange(hsv, low_range, upper_range)
    
    mask = cv2.erode(mask, skinkernel, iterations = 1)
    mask = cv2.dilate(mask, skinkernel, iterations = 1)
    
    #blur
    mask = cv2.GaussianBlur(mask, (15,15), 1)
    #cv2.imshow("Blur", mask)
    
    
    res = cv2.bitwise_and(roi, roi, mask = mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    return res

def binaryMask(frame, x0, y0, width, height ):    
#    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
#    roi = frame[y0:y0+height, x0:x0+width]
    
    #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame,(5,5),2)
    #blur = cv2.bilateralFilter(roi,9,75,75)
   
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #ret, res = cv2.threshold(blur, minValue, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)
    
    
    return res

cap = cv2.VideoCapture(0)


x0 = 100
y0 = 100
height = 300
width = 300

kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

i = 1

while(i):
    ret, frame = cap.read()
    if ret == True:
        
        roi = skinMask(frame, x0, y0, width, height, skinkernel)
        cv2.imshow('skin',roi)
        
        final = binaryMask(roi, x0, y0, width, height)
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
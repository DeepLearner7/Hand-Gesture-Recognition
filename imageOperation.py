#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:51:09 2017

@author: saurabh
"""

import cv2
import numpy
import matplotlib.pyplot as plt

img1 = cv2.imread('scene.jpg')
img2 = cv2.imread('planet.jpg')

add = img1+img2
cv2add = cv2.add(img1,img2)
weightAdd = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

cv2.imshow('add', add)
cv2.imshow('cv2add', cv2add)
cv2.imshow('weightAdd', weightAdd)


ret, threshold = cv2.threshold(img2, 12, 255, cv2.THRESH_BINARY)
cv2.imshow('threshold', threshold)
#cv2.adaptiveThreshold -> better one  

import os
import cv2 as cv
import numpy as np
img = cv.imread('123.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
harcascade = cv.CascadeClassifier('har_face.xml')
face_detect = harcascade.detectMultiScale(gray, scaleFactor= 1.1,minNeighbors=4)  

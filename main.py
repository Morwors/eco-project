import tensorflow as tf
import cv2 as cv2
import numpy as np
import vegetation

# print(cv.__version__)

# vegetation.getvegetation()
vegetation.createOverlay()

# image = cv2.imread('demo/img4.JPG')
#
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv, (40, 25, 25), (70, 255,255))
#
# cv2.imshow('Original', image)
# cv2.waitKey(0)
#
# imask = mask>0
# green = np.zeros_like(image, np.uint8)
# green[imask] = image[imask]
#
#
# cv2.imshow('Greenscale', green)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()


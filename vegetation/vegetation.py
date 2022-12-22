import sys

import numpy
import numpy as np
import cv2
import tensorflow as tf
import copy
import os
from os import listdir
from os.path import isfile, join

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def getvegetation(country):
    onlyfiles = [f for f in listdir('demo/' + country + '/imgs') if isfile(join('demo/' + country + '/imgs', f))]
    if len(onlyfiles) == 0:
        return False
    for file in onlyfiles:
        print("File: ", file)
        img = cv2.imread('demo/'+country+'/imgs/' + file)

        upperbound = np.array([70, 255, 255])
        lowerbound = np.array([40, 40, 40])
        mask = cv2.inRange(img, lowerbound, upperbound)
        imask = mask > 0
        white = np.full_like(img, [255, 255, 255], np.uint8)
        result = np.zeros_like(img, np.uint8)
        result[imask] = white[imask]

        image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Out', result)
        cv2.waitKey(0)

        blur = cv2.GaussianBlur(image, (9, 9), 0)
        _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('Out', thr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite('demo/' + country + '/whitemap/' + file, image)

    return True


def createOverlay(country):
    try:
        onlyfiles = [f for f in listdir('demo/' + country + '/whitemap') if
                     isfile(join('demo/' + country + '/whitemap', f))]
        if len(onlyfiles) == 0:
            return False
        for file in onlyfiles:
            colorImg = cv2.imread('demo/' + country + '/imgs/' + file)
            img = np.array(colorImg, dtype=np.float)
            img /= 255.0
            cv2.imshow('img', img)
            cv2.waitKey(0)

            whiteImg = cv2.imread('demo/' + country + '/whitemap/' + file)
            mask = np.array(whiteImg, dtype=np.float)
            mask /= 255.0
            transparency = .25
            mask *= transparency
            cv2.imshow('Mask', mask)
            cv2.waitKey(0)

            green = np.ones(img.shape, dtype=np.float) * (0, 1, 0)
            out = green * mask + img * (1.0 - mask)
            cv2.imshow('Out', out)
            cv2.waitKey(0)

            # cv2.imshow(winname='Overlay', mat=out)
            # cv2.waitKey(0)
            cv2.destroyAllWindows()

        return True
    except FileNotFoundError:
        print("File not found")
        raise FileNotFoundError
        # return False
    except:
        return False


def convertImgsToArrays(country):
    arrays = np.array([])
    onlyfiles = [f for f in listdir('demo/'+country+'/whitemap') if isfile(join('demo/'+country+'/whitemap', f))]
    for file in onlyfiles:
        print("Spinning files")
        img = cv2.imread('demo/'+country+'/whitemap/' + file, cv2.IMREAD_GRAYSCALE)
        img = img.astype(float)
        img = img / 255.0
        arrays = np.append(arrays, img)
    arrays = np.reshape(arrays, (-1, 944, 1570))
    return arrays
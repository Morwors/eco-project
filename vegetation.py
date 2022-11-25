import numpy as np
import cv2
import tensorflow as tf
import copy
import os
from os import listdir
from os.path import isfile, join

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def getvegetation():
    onlyfiles = [f for f in listdir('demo/bolivia/imgs') if isfile(join('demo/bolivia/imgs', f))]
    print("FIles: ", onlyfiles)
    for file in onlyfiles:
        print("File: ", file)
        img = cv2.imread('demo/bolivia/imgs/' + file)

        upperbound = np.array([70, 255, 255])
        lowerbound = np.array([40, 40, 40])
        # HSV UPPER (100,255,255) DOWN (40,36,25)
        mask = cv2.inRange(img, lowerbound, upperbound)
        # print(mask.shape)
        imask = mask > 0
        white = np.full_like(img, [255, 255, 255], np.uint8)
        result = np.zeros_like(img, np.uint8)
        result[imask] = white[imask]

        image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(image, (9, 9), 0)
        _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite('demo/bolivia/whitemap/' + file, thr)


def createOverlay():
    onlyfiles = [f for f in listdir('demo/bolivia/whitemap') if isfile(join('demo/bolivia/whitemap', f))]
    for file in onlyfiles:
        colorImg = cv2.imread('demo/bolivia/imgs/' + file)
        img = np.array(colorImg, dtype=np.float)
        img /= 255.0
        cv2.imshow('img', img)
        cv2.waitKey(0)

        whiteImg = cv2.imread('demo/bolivia/whitemap/' + file)
        mask = np.array(whiteImg, dtype=np.float)
        mask /= 255.0
        # set transparency to 25%
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


def pediction():
    start_img = cv2.imread('demo/bolivia/whitemap/bol1.jpg', cv2.IMREAD_GRAYSCALE)
    end_img = cv2.imread('demo/bolivia/whitemap/bol7.jpg', cv2.IMREAD_GRAYSCALE)

    xs = tf.convert_to_tensor(start_img, dtype=tf.float32)
    ys = tf.convert_to_tensor(end_img, dtype=tf.float32)
    # print(xs)
    #
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=32),
        # tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    #
    model.fit(xs, ys, epochs=500)
    #
    # results = model.predict(end_img)
    # print(results)

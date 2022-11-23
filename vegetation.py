import numpy as np
import cv2
import copy
from os import listdir
from os.path import isfile, join


def getvegetation():
    onlyfiles = [f for f in listdir('demo/china/imgs') if isfile(join('demo/china/imgs', f))]
    print("FIles: ", onlyfiles)
    for file in onlyfiles:
        print("File: ",file)
        img = cv2.imread('demo/china/imgs/'+file)

        upperbound = np.array([90, 255, 255])
        lowerbound = np.array([40, 36, 25])
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
        cv2.imwrite('demo/china/whitemap/'+file, thr)


def createOverlay():
    onlyfiles = [f for f in listdir('demo/china/whitemap') if isfile(join('demo/china/whitemap', f))]
    for file in onlyfiles:
        img = cv2.imread('demo/china/whitemap/'+file)
        fullimg = cv2.imread('demo/china/imgs/'+file)
        print("Shape: ", img.shape)
        mask = np.array(img, dtype=np.float)
        mask /= 255.0
        transparency = .25
        mask *= transparency
        overlayed = copy.deepcopy(fullimg)

        green = np.ones(overlayed.shape, dtype=np.float) * (0, 1, 0)
        print(green.shape, mask.shape, overlayed.shape)
        out = green * mask + fullimg * (1.0 - mask)
        print(type(img))

        # backtorgb = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)
        # backtorgb[np.all(backtorgb == (255, 255, 255), axis=-1)] = (0, 255, 0)
        # overlayed[thr == 255] = (36, 255, 12)
        # added_image = cv2.addWeighted(img, 0.4, overlayed, 0.1, 0)

        # cv2.imshow(winname='satellite image', mat=img)

        cv2.imshow(winname='Overlay', mat=out)
        # cv2.imshow('vegetation detection', result)

        # cv2.imshow('Blur', thr)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


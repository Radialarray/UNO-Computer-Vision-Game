import cv2
import numpy as np
import argparse
import glob
import time
import autocanny
import utils


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# Trainiertes Model in Datei speichern
from sklearn.externals import joblib
from sklearn.cluster import KMeans
import colorsys

from operator import itemgetter
from os import listdir
from os.path import isfile, join, isdir

# Webcam-Loop starten

cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_EXPOSURE, 40)
foundCards = []


while(cam.isOpened()):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bound_img = img
    blurVal = 7
    blurred = cv2.GaussianBlur(gray, (blurVal, blurVal), 0)

    ret, thresh = cv2.threshold(blurred, 220, 255, 0)
    im2, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("Thresh", thresh)

    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    auto = autocanny.auto_canny(thresh)

    # cv2.imshow("Edged", auto)

    # Find contours in edged imgs and convert to contour
    # im2,contours,hierarchy = cv2.findContours(auto, 1, 2)
    #cnt = contours[0]
    for idx, cnt in enumerate(contours):
        # M = cv2.moments(cnt)
        # epsilon = 0.1*cv2.arcLength(cnt,True)
        # approx = cv2.approxPolyDP(cnt,epsilon,True)
        # Draw contour on original img
        # cv2.drawContours(img, [cnt], -1, (0,255,0), 3)

        area = cv2.contourArea(cnt)
        # print('area: ' + str(area))
        if area > 5000 and area < 100000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            # print('AspectRatio: ' + str(aspect_ratio))

            if aspect_ratio > 0.8 and aspect_ratio < 1.5:
                # Draw bounding box

                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)


                bound_img = img[y:y + h, x:x + w]
                foundCards.append(bound_img)
                # idx = ind # The index of the contour that surrounds your object
                # Create mask where white is what we want, black otherwise
                mask = np.zeros_like(img)
                # Draw filled contour in mask
                cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
                # Extract out the object and place into output img
                out = np.zeros_like(img)
                out[mask == (255, 255, 255)] = img[mask == (255, 255, 255)]

                cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

                # Show the output img
                # windowname = "Mask: " + str(idx)
                # cv2.imshow(windowname, out)

                # show the imgs
                # cv2.imshow("Original", img)
                # cv2.imshow("Edges", np.hstack([wide, tight, auto]))
    cv2.imshow("ORIG", img)
    for idx, elem in enumerate(foundCards):
        elem = cv2.cvtColor(elem, cv2.COLOR_BGR2RGB)
        elem = elem.reshape((elem.shape[0] * elem.shape[1],3)) #represent as row*column,channel number
        clt = KMeans(n_clusters=3) #cluster number
        clt.fit(elem)


        hist = utils.find_histogram(clt)
        bar = utils.plot_colors2(hist, clt.cluster_centers_)
        # hsl = colorsys.rgb_to_hsv(bar[0][0][0],bar[0][0][1],bar[0][0][2])
        # print(bar[0])
        colors = clt.cluster_centers_
        print(colors)
        counter = 0
        for idx,row in enumerate(colors):
            for idc,elem in enumerate(row):
                # print(idx)
                # print(elem, end=' ')
                colors[idx][idc] = colors[idx][idc]/255
                counter += 1
            # print()

        saturate = []
        for idx,elem in enumerate(colors):
            # print(colors[idx][0])
            # print(colors[idx][1])
            # print('RGB: ' + str(colors))
            colors[idx] = colorsys.rgb_to_hsv(colors[idx][0],colors[idx][1],colors[idx][2])
            # print('HSV: ' + str(colors))
            saturate.append(colors[idx][1])
        # print(colors)
        # print(saturate)



        minRed = 0.9
        maxRed = 0.1

        minYellow = 0.1
        maxYellow = 0.3

        minGreen = 0.3
        maxGreen = 0.65

        minBlue = 0.65
        maxBlue = 0.9


        # Get the color with most saturation
        # print(colors[np.argmax(saturate)])

        # print(utils.findInRange(colors[np.argmax(saturate)][0],minGreen,maxGreen))



    foundCards[:] = []
    cv2.waitKey(1)

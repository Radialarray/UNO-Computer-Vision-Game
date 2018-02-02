import cv2
import numpy as np
import argparse
import glob
import time
import autocanny
import utils
#import outsourced functions from utils.py and autocanny.py

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# Trainiertes Model in Datei speichern
from sklearn.externals import joblib
from sklearn.cluster import KMeans
import colorsys

from operator import itemgetter
from os import listdir
from os.path import isfile, join, isdir

# Start webcam loop
cam = cv2.VideoCapture(1)

# Trying to set locked cam exposure with different values, not possible with webcam microsoft lifecam 3000 hd
cam.set(cv2.CAP_PROP_EXPOSURE, 40)

# Save found and cut out cards in an array for further processing
foundCards = []

# Starting processing loop
while(cam.isOpened()):

    # Read image
    ret, img = cam.read()

    # Convert image to gravscale image for edge and feature detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bound_img = img

    # Blur for filtering small details
    blurVal = 7
    blurred = cv2.GaussianBlur(gray, (blurVal, blurVal), 0)

    # Treshhold image for sorting out smaller objects and artifacts
    ret, thresh = cv2.threshold(blurred, 220, 255, 0)

    # Find contours in image, only external contours, sort inner contours out
    im2, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("Thresh", thresh)

    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    auto = autocanny.auto_canny(thresh)

    # Debug show edged image
    # cv2.imshow("Edged", auto)

    # Find contours in edged imgs and convert to contour
    # im2,contours,hierarchy = cv2.findContours(auto, 1, 2)
    #cnt = contours[0]

    # Loop over all found contours
    for idx, cnt in enumerate(contours):
        # M = cv2.moments(cnt)
        # epsilon = 0.1*cv2.arcLength(cnt,True)
        # approx = cv2.approxPolyDP(cnt,epsilon,True)
        # Draw contour on original img
        # cv2.drawContours(img, [cnt], -1, (0,255,0), 3)

        # Check for size of contour area, filter small and huge areas
        area = cv2.contourArea(cnt)
        # print('area: ' + str(area))
        if area > 5000 and area < 100000:
            # Get width and height of boundingrect of contour
            x, y, w, h = cv2.boundingRect(cnt)
            # Get aspect ratio of computed rectangle of found contour.
            # Filter for contours with the right image aspect ratio.
            aspect_ratio = float(w)/h
            # print('AspectRatio: ' + str(aspect_ratio))

            if aspect_ratio > 0.8 and aspect_ratio < 1.5:

                # Draw bounding box
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Compute values of bounding image and add it to the foundCards array.
                bound_img = img[y:y + h, x:x + w]
                foundCards.append(bound_img)
                # Create mask where white is what we want, black otherwise
                mask = np.zeros_like(img)
                # Draw filled contour in mask
                cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
                # Extract out the object and place into output img
                out = np.zeros_like(img)
                out[mask == (255, 255, 255)] = img[mask == (255, 255, 255)]

                # Draw contours onto image for debugging
                cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

                # Show all found cards in image as extra window
                # windowname = "Mask: " + str(idx)
                # cv2.imshow(windowname, out)

                # show the imgs
                # cv2.imshow("Original", img)
                # cv2.imshow("Edges", np.hstack([wide, tight, auto]))

    # Loop over all elements of foundCards
    # Enumerate is used to get an index value
    for idx, elem in enumerate(foundCards):
        # image element in foundCards gets converted to RGB colorspace
        elem = cv2.cvtColor(elem, cv2.COLOR_BGR2RGB)
        # represent as row*column,channel number
        elem = elem.reshape((elem.shape[0] * elem.shape[1],3))
        # cluster number for kmeans algorithm
        clt = KMeans(n_clusters=3)
        clt.fit(elem)


        # create histogramm with help of function in utils.py file.
        hist = utils.find_histogram(clt)

        # create bar plot of the found kmeans clusters
        bar = utils.plot_colors2(hist, clt.cluster_centers_)
        # hsl = colorsys.rgb_to_hsv(bar[0][0][0],bar[0][0][1],bar[0][0][2])
        # print(bar[0])

        # save colors as array for further processing
        colors = clt.cluster_centers_
        print(colors)
        counter = 0
        # loop over all elements of colors for converting values into right range for colorsys package
        for idx,row in enumerate(colors):
            for idc,elem in enumerate(row):
                # print(idx)
                # print(elem, end=' ')
                colors[idx][idc] = colors[idx][idc]/255
                counter += 1
            # print()

        # saturate is an array where we save all the saturation values for finding the color with the most saturation.
        # the kmeans cluster algorithm returns good results with three main colors detected in image and now it's time
        # to find the color value of an uno card which is the most saturated value.
        saturate = []
        for idx,elem in enumerate(colors):
            # print(colors[idx][0])
            # print(colors[idx][1])
            # print('RGB: ' + str(colors))
            colors[idx] = colorsys.rgb_to_hsv(colors[idx][0],colors[idx][1],colors[idx][2])
            # print('HSV: ' + str(colors))

            # append only the saturation values to the array
            saturate.append(colors[idx][1])
        # print(colors)
        # print(saturate)


        # color range values
        minRed = 0.9
        maxRed = 0.1

        minYellow = 0.1
        maxYellow = 0.3

        minGreen = 0.3
        maxGreen = 0.65

        minBlue = 0.65
        maxBlue = 0.9


        # Get the color index position with most saturation
        # print(colors[np.argmax(saturate)])

        # Get the color with the highest saturation.
        # utils.findInRange(colors[np.argmax(saturate)][0],minGreen,maxGreen)
        # print(utils.findInRange(colors[np.argmax(saturate)][0],minGreen,maxGreen))


    cv2.imshow("ORIG", img)
    # clear the foundCards array
    foundCards[:] = []
    cv2.waitKey(1)

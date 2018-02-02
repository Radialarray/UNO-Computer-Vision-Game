import cv2
import numpy as np
import argparse
import glob
import time
import autocanny

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Save trained model in file for faster startup
from sklearn.externals import joblib
from operator import itemgetter
from os import listdir
from os.path import isfile, join, isdir


# Mathematische "Dimensionalität" des ML-Algorithmus
dimensions = 128
hog = cv2.HOGDescriptor()

# load trained model to save startup time
clf = joblib.load('../objectDetection/feature_karten.pkl')

# start webcam loop
cam = cv2.VideoCapture(1)

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
    ret, thresh = cv2.threshold(blurred, 200, 255, 0)

    # Find contours in image, only external contours, sort inner contours out
    im2, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    auto = autocanny.auto_canny(thresh)

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
        if area > 1000 and area < 50000:
            # Get width and height of boundingrect of contour
            x, y, w, h = cv2.boundingRect(cnt)
            # Get aspect ratio of computed rectangle of found contour.
            # Filter for contours with the right image aspect ratio.
            aspect_ratio = float(w)/h

            if aspect_ratio > 0.5 and aspect_ratio < 2:

                # Draw bounding box
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (0, 0, 255), 2)


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

                # Show the output img
                # windowname = "Mask: " + str(idx)
                # cv2.imshow(windowname, out)

                # show the imgs
                # cv2.imshow("Original", img)
                # cv2.imshow("Edges", np.hstack([wide, tight, auto]))
    cv2.imshow("ORIG", img)

    # run the hogs algorithm for features from trained model, previously loaded from file.
    for idx, img in enumerate(foundCards):
        img_gray = cv2.resize(img, (dimensions, dimensions),
                              interpolation=cv2.INTER_AREA)
        features = hog.compute(img_gray).T
        print("Classification results:")
        print("=======================")
        print("Prediction for " + str(idx) + " image: " + str(clf.predict(features)))
    foundCards[:] = []
    cv2.waitKey(1)

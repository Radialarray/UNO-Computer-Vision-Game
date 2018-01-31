import cv2
import numpy as np

red = np.uint8([[[0,0,255 ]]])
hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
colorVal = "Red: " + str(hsv_red)
print (colorVal)

yellow = np.uint8([[[0,255,255 ]]])
hsv_yellow = cv2.cvtColor(yellow,cv2.COLOR_BGR2HSV)
colorVal = "Yellow: " + str(hsv_yellow)
print (colorVal)

green = np.uint8([[[0,255,0 ]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
colorVal = "Green: " + str(hsv_green)
print (colorVal)

blue = np.uint8([[[255,94,0 ]]])
hsv_blue = cv2.cvtColor(blue,cv2.COLOR_BGR2HSV)
colorVal = "Blue: " + str(hsv_blue)
print (colorVal)

purple = np.uint8([[[92,0,255 ]]])
hsv_purple = cv2.cvtColor(purple,cv2.COLOR_BGR2HSV)
colorVal = "Purple: " + str(hsv_purple)
print (colorVal)

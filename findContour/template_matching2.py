import cv2
import numpy as np

# read image
img = cv2.imread('images/3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('images/template.png',0)

# run template matching, get minimum val
res = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# create threshold from min val, find where sqdiff is less than thresh
min_thresh = (min_val + 1e-6) * 1.5
match_locations = np.where(res<=min_thresh)

# draw template match boxes
w, h = template.shape[::-1]
for (x, y) in zip(match_locations[1], match_locations[0]):
    cv2.rectangle(img, (x, y), (x+w, y+h), [0,255,255], 2)

# display result
cv2.imshow('', img)
cv2.waitKey(0)

import cv2
import numpy as np
def auto_canny(img, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(img)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(img, lower, upper)

	# return the edged img
	return edged

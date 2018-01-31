# import the necessary packages
import numpy as np
import argparse
import glob
import cv2


def auto_canny(img, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(img)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(img, lower, upper)

	# return the edged img
	return edged


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imgs", required=True,
	help="path to input dataset of imgs")
args = vars(ap.parse_args())

# loop over the imgs
for imgPath in glob.glob(args["imgs"] + "/*.jpg"):
	# load the img, convert it to grayscale, and blur it slightly
	img = cv2.imread(imgPath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)




	blurVal = 9
	blurred = cv2.GaussianBlur(gray, (blurVal, blurVal), 0)

	ret, thresh = cv2.threshold(blurred, 70, 255, 0)
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# cv2.imshow("Thresh", thresh)

	# apply Canny edge detection using a wide threshold, tight
	# threshold, and automatically determined threshold
	# wide = cv2.Canny(blurred, 10, 200)
	# tight = cv2.Canny(blurred, 225, 250)
	auto = auto_canny(thresh)

	cv2.imshow("Edged", auto)

	# Find contours in edged imgs and convert to contour
	# im2,contours,hierarchy = cv2.findContours(auto, 1, 2)
	#cnt = contours[0]
	for idx,cnt in enumerate(contours):
		# M = cv2.moments(cnt)
		# area = cv2.contourArea(cnt)
		# epsilon = 0.1*cv2.arcLength(cnt,True)
		# approx = cv2.approxPolyDP(cnt,epsilon,True)
	    # Draw contour on original img
		# cv2.drawContours(img, [cnt], -1, (0,255,0), 3)

	    # Draw bounding box
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(img,[box],0,(0,0,255),2)


		# idx = ind # The index of the contour that surrounds your object
		mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
		cv2.drawContours(mask, contours, idx, (255,255,255), -1) # Draw filled contour in mask
		out = np.zeros_like(img) # Extract out the object and place into output img
		out[mask == (255,255,255)] = img[mask == (255,255,255)]

		# Show the output img
		# windowname = "Mask: " + str(idx)
		# cv2.imshow(windowname, out)

		# show the imgs
		cv2.imshow("Original", img)
		# cv2.imshow("Edges", np.hstack([wide, tight, auto]))
	cv2.waitKey(0)

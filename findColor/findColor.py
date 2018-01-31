# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

# load the image
# image = cv2.imread(args["image"])
image = cv2.imread('../Spielkarten/Slices/farbe/Gelb/0_gelb.jpg', 1)

# define the list of boundaries
# rgb(255, 0, 0)
# rgb(255, 47, 0)
# rgb(255, 170, 0)
# rgb(242, 255, 0)
# rgb(221, 255, 0)
# rgb(150, 255, 0)
# rgb(0, 255, 140)
# rgb(0, 255, 240)
# rgb(13, 0, 255)
#RGB
#BGR
boundaries = [
	([0, 0, 255], [0, 47, 255]),
	([0, 255, 242], [0, 255, 221]),
	([0, 255, 150], [140, 255, 0]),
	([240, 255, 0], [255, 0, 13])
]


# boundaries = [
# 	([17, 15, 100], [50, 56, 200]),
# 	([86, 31, 4], [220, 88, 50]),
# 	([25, 146, 190], [62, 174, 250]),
# 	([103, 86, 65], [145, 133, 128])
# ]



# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)

	# show the images
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)

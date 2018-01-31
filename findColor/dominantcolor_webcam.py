# python color_kmeans.py --image images/jp.png --clusters 3

# Was steht als Scriptaufruf? : python dominantcolor.py -i 0_blau.jpg -c 1




# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2


cam = cv2.VideoCapture(0)

while(cam.isOpened()):
	# load the image and convert it from BGR to RGB so that
	# we can dispaly it with matplotlib
	ret, image = cam.read()
	image = cv2.pyrDown(image)
	image = cv2.pyrDown(image)
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# reshape the image to be a list of pixels
	image = image.reshape((image.shape[0] * image.shape[1], 3))

	# cluster the pixel intensities
	clt = KMeans(n_clusters = 1)
	clt.fit(image)

	# build a histogram of clusters and then create a figure
	# representing the number of pixels labeled to each color
	hist = utils.centroid_histogram(clt)
	bar = utils.plot_colors(hist, clt.cluster_centers_)

	bar[0][0][1] = 255
	domColor = bar[0][0]
	print(domColor)

	cv2.imshow("DOMCOLOR", bar)

	cv2.waitKey(1)

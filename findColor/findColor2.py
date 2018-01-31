import cv2
import numpy as np

image = cv2.imread('../Spielkarten/Slices/farbe/Blau/0_blau.jpg', 1)

# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

boundaries = [
	([169, 255, 255],[0,255,255]),
	([20, 255, 255], [50, 255, 255]),
	([50, 255, 255], [90, 255, 255]),
	([90, 255, 255], [120, 255, 255])
]

out_hsv = np.zeros((50,50,3),np.uint8)


print(cv2.mean(hsv))

out_hsv[:,:,0] = cv2.mean(hsv)[0]
out_hsv[:,:,1] = cv2.mean(hsv)[1]
out_hsv[:,:,2] = cv2.mean(hsv)[2]
print(out_hsv)
out_bgr = cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)
# print(out_bgr)
print(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HSV))
cv2.imshow("test",out_bgr)
cv2.waitKey(0)

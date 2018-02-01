import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

img = cv2.imread('rot.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
clt = KMeans(n_clusters=3) #cluster number
clt.fit(img)


hist = find_histogram(clt)
bar = plot_colors2(hist, clt.cluster_centers_)
hsl = colorsys.rgb_to_hsv(bar[0][0][0],bar[0][0][1],bar[0][0][2])
# print(bar[0])
colors = clt.cluster_centers_
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
    colors[idx] = colorsys.rgb_to_hsv(colors[idx][0],colors[idx][1],colors[idx][2])
    saturate.append(colors[idx][1])
# print(colors)
# print(saturate)



# thresh = 0.1
# minBGR =
# maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
#
# maskBGR = cv2.inRange(bright,minBGR,maxBGR)
# resultBGR = cv2.bitwise_and(bright, bright, mask = maskBGR)


# Get the color with most saturation
print(colors[np.argmax(saturate)])



plt.axis("off")
plt.imshow(bar)
plt.show()

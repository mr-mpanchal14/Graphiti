# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans


def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    d = []
    for (percent, color) in zip(hist, centroids):
        d.append([percent, color])
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
    return bar, d

image = cv2.imread(r'C:\Users\mannp\Desktop\Pycharm\Minor Project\P2.png')
image = image.reshape((image.shape[0] * image.shape[1], 3))
clt = KMeans(n_clusters = 5)
clt.fit(image)
hist = centroid_histogram(clt)
bar, dic = plot_colors(hist, clt.cluster_centers_)
bar = cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)
dic.sort()
dic = dic[:-1]
s = sum([x[0] for x in dic])

#calculate the % of the cluster
for i in range(len(dic)):
    dic[i][0] = 100*dic[i][0]/s
    dic[i][1] = [int(x) for x in dic[i][1]]

for i in dic:
    print(i)


plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()

bar = np.zeros((50, 300, 3), dtype="uint8")

# startX = 0
# for i in dic:
#     endX = startX + (i[0] * 300)
#     cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
#                   i[1], -1)
#     startX = endX
#
# plt.figure()
# plt.axis("off")
# plt.imshow(bar)
# plt.show()
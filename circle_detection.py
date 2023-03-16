import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image.
img = cv2.imread(r'D:\Users\Nishq\Desktop\P2.png', cv2.IMREAD_COLOR)

# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.medianBlur(gray, 25) #cv2.bilateralFilter(gray,10,50,50)

minDist = 100
param1 = 35 #500
param2 = 20 #200 #smaller value-> more false circles
minRadius = 100
maxRadius = 200 #10

# docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

# Show result for testing:
plt.imshow(img)
plt.show()
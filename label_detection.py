import cv2
import re
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from matplotlib import rcParams
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\mannp\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def findMaxConsecutiveOnes(nums) -> int:
    count = maxCount = 0
    for i in range(len(nums)):
        if nums[i] == 1:
            count += 1
        else:
            maxCount = max(count, maxCount)
            count = 0
    return max(count, maxCount)

def detectAxes(filepath, threshold=None, debug=False):
    if filepath is None:
        return None, None
    if threshold is None:
        threshold = 10
    image = cv2.imread(filepath)
    height, width, channels = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    maxConsecutiveOnes = [findMaxConsecutiveOnes(gray[:, idx] < 200) for idx in range(width)]
    start_idx, maxindex, maxcount = 0, 0, max(maxConsecutiveOnes)
    while start_idx < width:
        if abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
            maxindex = start_idx
            break
        start_idx += 1
    yaxis = (maxindex, 0, maxindex, height)
    if debug:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].plot(maxConsecutiveOnes, color='k')
        ax[1].axhline(y=max(maxConsecutiveOnes) - 10, color='r', linestyle='dashed')
        ax[1].axhline(y=max(maxConsecutiveOnes) + 10, color='r', linestyle='dashed')
        ax[1].vlines(x=maxindex, ymin=0.0, ymax=maxConsecutiveOnes[maxindex], color='b', linewidth=4)
        plt.show()
    maxConsecutiveOnes = [findMaxConsecutiveOnes(gray[idx, :] < 200) for idx in range(height)]
    start_idx, maxindex, maxcount = 0, 0, max(maxConsecutiveOnes)
    while start_idx < height:
        if abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
            maxindex = start_idx
        start_idx += 1
    cv2.line(image, (0, maxindex), (width, maxindex), (255, 0, 0), 2)
    xaxis = (0, maxindex, width, maxindex)
    if debug:
        rcParams['figure.figsize'] = 15, 8
        fig, ax = plt.subplots(1, 1)
        ax.imshow(image, aspect='auto')
    return xaxis, yaxis

def getTextFromImage(filepath, bw=False, debug=False):
    image_text = []
    image = cv2.imread(filepath)
    height, width, _ = image.shape
    if bw:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_val = np.array([0, 0, 0])
        upper_val = np.array([179, 255, 179])
        mask = cv2.inRange(hsv, lower_val, upper_val)
        image = cv2.bitwise_not(mask)
    d = pytesseract.image_to_data(image, config="-l eng --oem 1 --psm 11", output_type=Output.DICT)
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if float(d['conf'][i]) >= 0:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            image_text.append((d['text'][i], (x, y, w, h)))
    if bw:
        image = cv2.imread(filepath)
        image_text = list(set(image_text))
        white_bg = 255 * np.ones_like(image)
        for text, (textx, texty, w, h) in image_text:
            roi = image[texty:texty + h, textx:textx + w]
            white_bg[texty:texty + h, textx:textx + w] = roi
        image_text = []
        d = pytesseract.image_to_data(white_bg, config="-l eng --oem 1 --psm 11", output_type=Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if float(d['conf'][i]) >= 0:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                image_text.append((d['text'][i], (x, y, w, h)))
    return list(set(image_text))


def getProbableLabels(image, image_text, xaxis, yaxis):
    x_labels = []
    legends = []
    height, width, channels = image.shape
    for text, (textx, texty, w, h) in image_text:
        text = text.strip()
        (x1, y1, x2, y2) = xaxis
        (x11, y11, x22, y22) = yaxis
        if (np.sign((x2 - x1) * (texty - y1) - (y2 - y1) * (textx - x1)) == 1 and
                np.sign((x22 - x11) * (texty - y11) - (y22 - y11) * (textx - x11)) == -1):
            x_labels.append((text, (textx, texty, w, h)))
        elif (np.sign((x2 - x1) * (texty - y1) - (y2 - y1) * (textx - x1)) == -1 and
              np.sign((x22 - x11) * (texty - y11) - (y22 - y11) * (textx - x11)) == -1):
            if not bool(re.findall(r'\b[\d\.\d]+\b', text)):
                legends.append((text, (textx, texty, w, h)))
    maxIntersection = 0
    maxList = []
    for i in range(y1, height):
        count = 0
        current = []
        for index, (text, rect) in enumerate(x_labels):
            if lineIntersectsRectY(i, rect):
                count += 1
                current.append(x_labels[index])

        if count > maxIntersection:
            maxIntersection = count
            maxList = current
    def getYFromRect(item):
        return item[1]
    maxList.sort(key=getYFromRect)
    x_labels = []
    for text, (textx, texty, w, h) in maxList:
        x_labels.append(text)
        cv2.rectangle(image, (textx, texty), (textx + w, texty + h), (255, 0, 0), 2)
    maxIntersection = 0
    maxList = []
    for i in range(y1):
        count = 0
        current = []
        for index, (text, rect) in enumerate(legends):
            if lineIntersectsRectY(i, rect):
                count += 1
                current.append(legends[index])

        if count > maxIntersection:
            maxIntersection = count
            maxList = current
    for i in range(x11, width):
        count = 0
        current = []
        for index, (text, rect) in enumerate(legends):
            if lineIntersectsRectX(i, rect):
                count += 1
                current.append(legends[index])

        if count > maxIntersection:
            maxIntersection = count
            maxList = current
    legends = []
    legendBoxes = []
    for text, (textx, texty, w, h) in maxList:
        legends.append(text)
        legendBoxes.append((textx, texty, w, h))
    legendBoxes = mergeRects(legendBoxes)
    for (textx, texty, w, h) in legendBoxes:
        cv2.rectangle(image, (textx, texty), (textx + w, texty + h), (255, 0, 255), 2)
    print("number of clusters : ", len(legendBoxes))
    return image, x_labels, 0, legends

def lineIntersectsRectX(candx, rect):
    (x, y, w, h) = rect
    if x <= candx <= x + w:
        return True
    else:
        return False

def lineIntersectsRectY(candy, rect):
    (x, y, w, h) = rect
    if y <= candy <= y + h:
        return True
    else:
        return False

def getTextFromImageArray(image, mode):
    image_text = []
    if mode == 'y-text':
        image = cv2.transpose(image)
        image = cv2.flip(image, flipCode=1)
        config = "-l eng --oem 1 --psm 11"
    elif mode == 'y-labels':
        config = "-l eng --oem 1 --psm 6 -c tessedit_char_whitelist=.0123456789"
    d = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if float(d['conf'][i]) >= 0:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            image_text.append((d['text'][i], (x, y, w, h)))
    return list(set(image_text))

def maskImageForwardPass(filepath, start_idx):
    image = cv2.imread(filepath)
    height, width, channels = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    start_idx = 1
    while start_idx <= width:
        if sum(gray[:, start_idx] < 200) != 0:
            break
        else:
            start_idx += 1
    end_idx = start_idx
    while end_idx <= width:
        if sum(gray[:, end_idx] < 200) == 0:
            break
        else:
            end_idx += 1
    gray[:, 1:start_idx] = 255
    gray[:, end_idx:width] = 255
    return gray, start_idx, end_idx

def maskImageBackwardPass(filepath, end_idx):
    image = cv2.imread(filepath)
    height, width, channels = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    while end_idx > 0:
        if sum(gray[:, end_idx] < 200) == 0:
            break
        else:
            end_idx -= 1
    gray[:, end_idx:width] = 255
    return gray

def nearbyRectangle(current, candidate, threshold):
    (currx, curry, currw, currh) = current
    (candx, candy, candw, candh) = candidate
    currymin = curry
    currymax = curry + currh
    candymin = candy
    candymax = candy + candh
    if candymax <= currymin <= candymax + threshold:
        return True
    if currymax <= candymin <= currymax + threshold:
        return True
    if candymax >= currymin >= candymin:
        return True
    if currymax >= candymin >= currymin:
        return True
    if currymin <= candymin <= currymax and currymin <= candymax <= currymax:
        return True
    return False


def mergeRects(contours):
    rects = []
    rectsUsed = []
    for cnt in contours:
        rects.append(cnt)
        rectsUsed.append(False)
    def getXFromRect(item):
        return item[0]
    rects.sort(key=getXFromRect)
    acceptedRects = []
    xThr = 5
    yThr = 5
    for supIdx, supVal in enumerate(rects):
        if (rectsUsed[supIdx] == False):
            currxMin = supVal[0]
            currxMax = supVal[0] + supVal[2]
            curryMin = supVal[1]
            curryMax = supVal[1] + supVal[3]
            rectsUsed[supIdx] = True
            for subIdx, subVal in enumerate(rects[(supIdx + 1):], start=(supIdx + 1)):
                candxMin = subVal[0]
                candxMax = subVal[0] + subVal[2]
                candyMin = subVal[1]
                candyMax = subVal[1] + subVal[3]
                if (candxMin <= currxMax + xThr):
                    if not nearbyRectangle((candxMin, candyMin, candxMax - candxMin, candyMax - candyMin),
                                           (currxMin, curryMin, currxMax - currxMin, curryMax - curryMin), yThr):
                        break
                    currxMax = candxMax
                    curryMin = min(curryMin, candyMin)
                    curryMax = max(curryMax, candyMax)
                    rectsUsed[subIdx] = True
                else:
                    break
            acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])
    return acceptedRects


def getProbableYLabels(image, contours, xaxis, yaxis):
    (x11, y11, x22, y22) = yaxis
    maxIntersection = 0
    maxList = []
    for i in range(x11):
        count = 0
        current = []
        for index, rect in enumerate(contours):
            if lineIntersectsRectX(i, rect):
                count += 1
                current.append(contours[index])
        if count > maxIntersection:
            maxIntersection = count
            maxList = current
    return image, maxList

def getYFromRect(item):
    return item[1][1]


def getXFromRect(item):
    return item[1][0]


filepath = r'C:\Users\mannp\Desktop\Pycharm\Minor Project\Output.png'
image = cv2.imread(filepath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, channels = image.shape
xaxis, yaxis = detectAxes(filepath)
y_text, y_labels = [], []

for (x1, y1, x2, y2) in [xaxis]:
    xaxis = (x1, y1, x2, y2)

for (x1, y1, x2, y2) in [yaxis]:
    yaxis = (x1, y1, x2, y2)
rcParams['figure.figsize'] = 15, 4
fig, ax = plt.subplots(1, 3)
gray = maskImageBackwardPass(filepath, yaxis[0])
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
thresh = cv2.dilate(thresh, rect_kernel, iterations = 1)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
thresh = cv2.dilate(thresh, rect_kernel, iterations = 1)
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
rects = [cv2.boundingRect(contour) for contour in contours]
print("number of contours: ", len(contours))

image_text = getTextFromImage(filepath, bw=True)
image, y_labels = getProbableYLabels(image, rects, xaxis, yaxis)
white_bg = 255 * np.ones_like(gray.copy())
for (textx, texty, w, h) in y_labels:
    roi = gray[texty:texty + h, textx:textx + w]
    white_bg[texty:texty + h, textx:textx + w] = roi
y_labels_list = getTextFromImageArray(white_bg, 'y-labels')
ax[0].imshow(white_bg, aspect = 'auto')
y_labels_list.sort(key = getYFromRect)
y_labels = []
for text, (textx, texty, w, h) in y_labels_list:
    roi = 255 * np.ones_like(gray[texty:texty + h, textx:textx + w])
    gray[texty:texty + h, textx:textx + w] = roi
    y_labels.append(text)
y_text_list = getTextFromImageArray(gray, 'y-text')
y_text_list.sort(key = getXFromRect)
for text, (textx, texty, w, h) in y_text_list:
    y_text.append(text)
image_text = getTextFromImage(filepath, bw=True)
image, x_labels, _, legends = getProbableLabels(image, image_text, xaxis, yaxis)

print("x-labels     :  ", x_labels)
print("y-text       :  ", y_text)
print("y-labels     :  ", y_labels)
print("legends      :  ", legends, end = "\n\n")
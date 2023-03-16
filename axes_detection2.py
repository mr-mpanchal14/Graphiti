#FINAL BAR CHART

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tkinter import * #for GUI
from PIL import Image, ImageTk
import pytesseract
from matplotlib import rcParams
from tkinter import filedialog
from pytesseract import Output
from sklearn.linear_model import LinearRegression
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\mannp\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


o = [0, 0]


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
    global image
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
    cv2.line(image, (maxindex, 0), (maxindex, height), (0, 0, 255), 2)
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

#contours are sub parts of an image
#this function is used to extact textbox from contours
def getTextFromImage(filepath, bw=False, debug=False):
    image_text = []
    image = cv2.imread(filepath)
    #image shape is identified and stored in variables height and weight
    height, width, _ = image.shape

    #using pytessract, save text information from contours of an image
    d = pytesseract.image_to_data(image, config="-l eng --oem 1 --psm 11", output_type=Output.DICT)
    #a pytessract object has 5 properties namely height,width,left,top,text. Which is stored in tuples
    #and is further stored in image_text
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if float(d['conf'][i]) >= 0:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            image_text.append((d['text'][i], (x, y, w, h)))

    return list(set(image_text))

#function to identify what text is label or a part of legend
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
        if rectsUsed[supIdx] == False:
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
                if candxMin <= currxMax + xThr:
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

def x_axis_change(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.line(prev, (0,y), (image.shape[1], y), (0, 0, 225), 2)
        o[1] = y

def y_axis_change(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.line(prev, (x, 0), (x, image.shape[0]), (225, 0, 0), 2)
        o[0] = x

#image upload panel in tkinter
def select_image():
    global pa, pb, prev, image, height, width, channels, xaxis, yaxis, path, original

    path = filedialog.askopenfilename() #to extract image from a path
    if len(path) > 0: #if File path is not None, image is extracted
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prev = image.copy() #save original image copy
        original = image.copy()
        height, width, channels = image.shape
        xaxis, yaxis = detectAxes(path) # function call to extract axis

    im1 = prev.copy()
    im2 = image.copy()
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB) # convert Blue Green Red to Red green Blue format
    im1 = Image.fromarray(im1) # convert into Tkinter Image format
    im1 = ImageTk.PhotoImage(im1) #convert into displayable image
    im2 = Image.fromarray(im2) # convert tinto Tkinter image format
    im2 = ImageTk.PhotoImage(im2) # convert into displayable image
    if pa is None or pb is None: # if both image exists
        pa = Label(image=im1) # defining Image Label
        pa.image = im1 # giving image values to image propoerty
        pa.pack(side="left", padx=10, pady=10) # defining size of the image
        pb = Label(image=im2) # defining label
        pb.image = im2 # give image values
        pb.pack(side="right", padx=10, pady=10) #defining size
    else:
        pa.configure(image=im1) #initialising None for no image
        pb.configure(image=im2)# initialising None for no image
        pa.image = im1 # giving image values if path is read
        pb.image = im2 # giving image values if path is read

win = Tk() #calling TKinter window
pa = pb = None # initialising image panels

ch = "" #initialising choiice Variable

btn = Button(win, text="Select an image", command=select_image) #call the Tkinter Panel
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10") #display tkinter panel
def btn1(): # command function for Confirm
  global ch
  ch = "C" #update the value of ch variable
  win.destroy() #close the window
button1 = Button(win, text="Confirm", command=btn1) #initialise Confiirm Button
button1.place(anchor=SE) # Placement of Confirm Button
button1.pack(side="bottom", fill="both", padx="18", pady="4") #definign size of Confirm Button

def btn2():
  global ch
  ch = "X"
  win.destroy()
button2 = Button(win, text="Edit Axes", command=btn2)
button2.place(anchor=SW)
button2.pack(side="bottom", fill="both", padx="18", pady="4")
win.mainloop() # run tkinter
y_text, y_labels = [], [] #initialise variables

for (x1, y1, x2, y2) in [xaxis]: # finding  x axis
    xaxis = (x1, y1, x2, y2)

for (x1, y1, x2, y2) in [yaxis]: # finding y axis
    yaxis = (x1, y1, x2, y2)

# print(xaxis, yaxis)

blank_image = np.zeros((100, image.shape[1], 3), np.uint8) #creating blank image to append below
#displaying text on the blank image
cv2.putText(blank_image, "Press U for Undo the Change", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.putText(blank_image, "Press C for Confirming the Change", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
blank_image2 = blank_image.copy() #making a copy of blank image
cv2.putText(blank_image, "Click to select X axis", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.putText(blank_image2, "Click to select Y axis", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#concatenate original image with info image
prev = cv2.vconcat([prev, blank_image])

if ch == 'X': # for setting manual axes
    image = prev.copy()
    while 1:
        cv2.imshow('changex', prev)
        cv2.setMouseCallback('changex', x_axis_change)
        k = cv2.waitKey(33)
        if k == ord('c'):
            break
        elif k == ord('u'):
            prev = image.copy()
    cv2.destroyAllWindows()
    prev = prev[:-100, :]
    prev = cv2.vconcat([prev, blank_image2])
    image = prev.copy()
    while 1:
        cv2.imshow('changey', prev)
        cv2.setMouseCallback('changey', y_axis_change)
        k = cv2.waitKey(33)
        if k == ord('c'):
            break
        elif k == ord('u'):
            print('undo')
            prev = image.copy()
    cv2.destroyAllWindows()
    image = prev.copy()
image = image[:-100, :]
if o[0]+o[1] !=0:
    xaxis = (xaxis[0], o[1], xaxis[2], o[1])
    yaxis = (o[0], yaxis[1], o[0], yaxis[3])
    # print(xaxis, yaxis)

rcParams['figure.figsize'] = 15, 4 # display size of image
fig, ax = plt.subplots(1, 3) #make a blank plot
gray = maskImageBackwardPass(path, yaxis[0]) #configure image masking according to axes
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) #threshholding
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)) #Morphing image
thresh = cv2.dilate(thresh, rect_kernel, iterations=1) #Dilating the Image
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)) #again Morphing the image
thresh = cv2.dilate(thresh, rect_kernel, iterations=1) # again dilating the image
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #finding contours
contours = contours[0] if len(contours) == 2 else contours[1] #finding best contours
rects = [cv2.boundingRect(contour) for contour in contours]
print("number of contours: ", len(contours)) #displaying no. of contours

image_text = getTextFromImage(path, bw=True) # function call to extracct text
image, y_labels = getProbableYLabels(image, rects, xaxis, yaxis) #function call to get y labels
white_bg = 255 * np.ones_like(gray.copy()) # creating blank image
for (textx, texty, w, h) in y_labels: # extracting region of interests, ignoring the noise
    roi = gray[texty:texty + h, textx:textx + w]
    white_bg[texty:texty + h, textx:textx + w] = roi
y_labels_list = getTextFromImageArray(white_bg, 'y-labels') # Actual Text extraction
ax[0].imshow(white_bg, aspect='auto') #show roi image
y_labels_list.sort(key=getYFromRect) #sort textboxs on the basis of position
y_labels = []
for text, (textx, texty, w, h) in y_labels_list: #extract ROI for Y labels
    roi = 255 * np.ones_like(gray[texty:texty + h, textx:textx + w])
    gray[texty:texty + h, textx:textx + w] = roi
    y_labels.append(text)
y_text_list = getTextFromImageArray(gray, 'y-text') #actual text extraction
y_text_list.sort(key=getXFromRect) #sort on the basis of X position

dy = []
dx = []
for i in y_labels_list: #extract y labels
    dy.append(int(i[0]))
    dx.append((i[1][1] + i[1][3]))


dx = np.array(dx).reshape((-1, 1)) #reshaping
for text, (textx, texty, w, h) in y_text_list:
    y_text.append(text)
image_text = getTextFromImage(path, bw=True)
image, x_labels, _, legends = getProbableLabels(image, image_text, xaxis, yaxis)

image = original.copy() #copy original image
blank_image = np.zeros((250, image.shape[1], 3), np.uint8)
# put texts in image
cv2.putText(blank_image, f"x-labels :  {x_labels}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.putText(blank_image, f"y-texts  :  {y_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.putText(blank_image, f"y-labels :  {y_labels}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.putText(blank_image, f"legends  :  {legends}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.putText(blank_image, "Press Q to Quit", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

try:
    # put text on image
    cv2.putText(blank_image, "Click Anywhere to display the Coordinate", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    image = cv2.vconcat([image, blank_image]) #concat image
    model = LinearRegression().fit(dx, dy) # use linear regression model to correlate pixel value to extracted value in x co-ordinate
    xc = model.intercept_ # store interecept of regressed model
    xm = model.coef_ #store slope of regressed model

    def display_coordinates(event, x, y, flags, param): # callback mouse event for displaying co-ordinates
        global image
        if event == cv2.EVENT_LBUTTONDOWN: # if triggered left click
            image = image[:-40, :]
            y = o[1] + y
            temp = np.zeros((40, image.shape[1], 3), np.uint8) #create blank image
            ans = xm*y + xc #extract co-oridnate
            cv2.putText(temp, f"Y Value: {ans.round(2)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) #put text
            image = cv2.vconcat([image, temp]) #concatenate images

    while 1:
        cv2.imshow('display_coordinates', image) #display image
        cv2.setMouseCallback('display_coordinates', display_coordinates) # set mouse call back event
        k = cv2.waitKey(33) # track keyboard
        if k == ord('q'): #if pressed 'q'
            break
except:
    cv2.putText(blank_image, "Couldn't Detect Y labels", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    image = cv2.vconcat([image, blank_image])
    while 1:
        cv2.imshow('display_coordinates', image)
        k = cv2.waitKey(33)
        if k == ord('q'):
            break


            
from asyncio.windows_events import NULL
import cv2 as cv
from cv2 import threshold
from cv2 import waitKey
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import empty

#hsv green range
lower_green = np.array([40, 40, 40])
upper_green = np.array([70, 255, 255])

img =cv.imread("sayac_projesi/sayacyesil.jpg")  # read image

#mask image
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # convert to hsv
blur = cv.GaussianBlur(hsv_img, (5, 5), 0)  # blur
mask = cv.inRange(blur, lower_green, upper_green)  # mask


_, threshold = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)  # threshold
contours = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # find contours
contours = sorted(contours[0], key=cv.contourArea, reverse=True)  # sort contours

arr = np.array(NULL)  # array

for i in contours[:4]:  # find 4 largest contours
    x,y,w,h = cv.boundingRect(i)
    cv.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
    print(x,y,w,h)
    liste = [x,y]
    arr = np.append(arr, [liste])

#perpective transform
width, height = img.shape[:2]
pts1 = np.float32([[arr[1],arr[2]], [arr[3],arr[4]], [arr[5],arr[6]], [arr[7],arr[8]]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix  = cv.getPerspectiveTransform(pts1,pts2)
img_out  = cv.warpPerspective(img,matrix,(width,height))
out=cv.rotate(img_out, cv.ROTATE_90_COUNTERCLOCKWISE)

#------------------
out  = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
_, threshold_out = cv.threshold(out, 125, 255, cv.THRESH_BINARY_INV)
contours = cv.findContours(threshold_out, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours[0], key=cv.contourArea, reverse=False)

for a in contours:
    x,y,w,h = cv.boundingRect(a)
    cv.rectangle(threshold_out, (x, y), (x + w, y + h), (255,0,255), 2)
    break

#todo:
#1. kus bakısı baktıgın için butun sayacları rakanmların yerler aynı olucak ondan dolayı dırek kırpabilirsin
#2. kırpımıs yerleri sayıları tek tek tahminde bulunması lazım 
#3. sayı tahmin algoritması geliştirilmeli ve araştırılmalı (SVGLinearRegression) vb.




cv.imshow("thresh", threshold_out)
cv.imshow("img", img)
print("array: ", arr)

cv.imshow("img_out", out)

waitKey(0)
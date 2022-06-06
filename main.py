from cv2 import blur
import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt
import io


img = cv.imread("sayac3.jpg",1 )  # read image
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert to gray
img_filter = cv.medianBlur(img_hsv,7)  # filter image
blur_gauss = cv.GaussianBlur(img_filter,(5,5),0)  # blur image
img_blur = cv.blur(blur_gauss,(5,5))  # blur image
img_filter2D = cv.filter2D(img_blur,-1,np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))  # filter image




# tresholding
_ ,thresh = cv.threshold(img_filter2D,125,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)

#contours
contours, _ = cv.findContours(close,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key = cv.contourArea, reverse = True)[:10]


min_area = 3000
max_area = 5000
image_number = 0
for c in contours:
    area = cv.contourArea(c)
    if area > min_area and area < max_area:
        x,y,w,h = cv.boundingRect(c)
        ROI = img[y:y+h, x:x+w]
        cv.imwrite('ROI_{}.png'.format(image_number), ROI)
        cv.rectangle(img, (x, y), (x + w, y + h), (255,255,0), 2)
        image_number += 1



cv.imshow("conturs",img)
cv.imshow("thresh",thresh)

cv.waitKey(0)
cv.destroyAllWindows()




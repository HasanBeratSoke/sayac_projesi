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

img =cv.imread("sayac_projesi/data/sayacyesil.jpg")  # read image

#mask image
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # convert to hsv
blur = cv.GaussianBlur(hsv_img, (5, 5), 0)  # blur
mask = cv.inRange(blur, lower_green, upper_green)  # mask

#threshold image and contour
_, threshold = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)  # threshold
contours = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # find contours
contours = sorted(contours[0], key=cv.contourArea, reverse=True)  # sort contours

#yeşil noktaların koordinatlarını tutulduğu liste
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

#----------------------------------------------------------------------------------------------------------------------

#todo:
#1. kus bakısı baktıgın için butun sayacları rakanmların yerler aynı olucak ondan dolayı dırek kırpabilirsin
#2. kırpımıs yerleri sayıları tek tek tahminde bulunması lazım 
#3. sayı tahmin algoritması geliştirilmeli ve araştırılmalı (SVGLinearRegression) vb.



# koordinatlar top-left(188,275)  bottom-rigth(264,410) burdan baslayyarak x eksnini 125 picsel arttırarak sayılara böl 
# 115-197 -- 930-342

#sayıların ldugu kısım alındı kesildi
cv.rectangle(out, (115,197),(930,342), (0,0,255), 3)
out  = out[197:342, 115:930]
out = cv.GaussianBlur(out, (5, 5), 0)
out = cv.cvtColor(out, cv.COLOR_HSV2BGR)
out = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
_, out = cv.threshold(out, 90, 210, cv.THRESH_BINARY_INV)




#112-32 = 80
# üst sol(3,9)--alt sağ(110-115)
# 95 boyutlu bir butu olması lazım x ekseni



cv.imshow("img_out", out)
cv.imwrite("sayac_projesi/sonuc.jpg", out)

waitKey(0)
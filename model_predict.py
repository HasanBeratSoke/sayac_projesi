from multiprocessing.connection import wait
from keras.datasets import mnist
from cv2 import destroyAllWindows
import numpy as np
import cv2 as cv
from skimage import img_as_ubyte    
from skimage.color import rgb2gray
from keras.models import load_model

img =cv.imread('sayac_projesi/sonuc.jpg')

model = load_model('sayac_projesi/trained_model.h5') 

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img_gray_u8 = img_as_ubyte(img_gray) # convert to uint8

(thresh, im_binary) = cv.threshold(img_gray_u8, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) # thresholding

img_resized = cv.resize(im_binary,(28,28)) # resize to 28x28

im_gray_invert = 255 - img_resized  # invert the image

cv.imshow("invert image", im_binary)   # show the image

im_final = im_gray_invert.reshape(1,28,28,1) # reshape the image to 4D array
ans = model.predict(im_final) # predict the image
ans = np.argmax(ans,axis=1)[0] # get the index of the max value
print(ans)

cv.putText(im_binary,'Predicted Digit : '+str(ans),
                    (50,50),cv.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)
  
cv.imshow("Original Image",im_binary)


(x_train, y_train),(x_test, y_test) = mnist.load_data()
 
import matplotlib.pyplot as plt
fig, axes = plt.subplots(10, 10, figsize=(8, 8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(y_train[i]),transform=ax.transAxes, color='green')
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
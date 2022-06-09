
from pydoc import importfile
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import keras as kr
import tensorflow as tf

importfile('sayac_projesi/perspective.py')
importfile('sayac_projesi/digit_recog.py')

class DigitRecog:
    def __init__(self):
        self.model = kr.models.load_model('sayac_projesi/model.h5')
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        self.model.load_weights('sayac_projesi/weights.h5')
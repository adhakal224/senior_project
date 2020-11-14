#!/usr/bin/env python
# coding: utf-8

# In[59]:


import os
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as mplot
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#loads the monet images and regular images as numpy array and returns them as attributes of the file_load object
class file_load(object):
    
    def __init__(self):
        self.start = True
    
    
    def load_images(self, path, size=(256,256)):
        X_images = []
        for image in os.listdir(path):
            new_img = load_img(path + image, target_size=size)
            img_array = img_to_array(new_img)
            X_images.append(img_array)
        X_images = np.asarray(X_images)/255
        return X_images
    
    def get_img(self, path):
        with tf.device('/GPU:0'):
            img_array = self.load_images(path, (256,256))
            return img_array


# In[ ]:





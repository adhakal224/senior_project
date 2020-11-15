#!/usr/bin/env python
# coding: utf-8

# In[22]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mplot
import cv2


# In[23]:


import gan_load as gload


# In[139]:


import sys


# In[24]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# In[26]:


def get_generator(path):
    generator = tf.keras.models.load_model(path)
    
    return generator


# In[28]:


def get_img(path):
    loader = gload.file_load()
    img_array = loader.get_img(path)
    
    return img_array


# In[29]:


def squeeze_img(img):
    img = np.squeeze(img, axis=0)
    return img


# In[30]:


def de_normalize(img):
    img = (img + 1)/2
    return img


# In[131]:


def show_img(img):
    mplot.imshow(img)
    mplot.axis('off')
    mplot.show()


# In[68]:


def pre_process(img):
    img = (tf.cast(img, tf.float32) / 2) - 1
    return img


# In[118]:


def generate_art(generator_path, img_path):
    
    generator_monet = get_generator(generator_path)
    
    img_array = get_img(img_path)
    
    img_array = (tf.cast(img_array, tf.float32)*2) -1

    generated_img = generator_monet(img_array)

    my_img = de_normalize(img_array)
    gen_img = de_normalize(generated_img)

    
    return my_img, gen_img 


# In[136]:


def save_to_file(path, gen_img):
    mplot.imshow(gen_img)
    mplot.axis('off')
    mplot.savefig(path)


	
# In[138]:


if __name__ == '__main__':
	monet_gen_path = sys.argv[1] + '/'
	img_path = sys.argv[2] + '/'
	save_path = sys.argv[3] + '/'
	my_img, gen_img = generate_art(monet_gen_path, img_path)

	for epoch, image in enumerate(gen_img):
		temp_path = save_path + '{}.png'.format(epoch)
		save_to_file(temp_path, image)


   

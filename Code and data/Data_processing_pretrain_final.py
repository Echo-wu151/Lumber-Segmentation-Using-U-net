#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('''
----------------------------------------------------------------------------------
This file is Data_processing_pretrain_final.py

Importing functions.......
''')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
import sys
from draw_label_final import *


# In[3]:


# read the image with gray and store it into a dictionary 
# for image and original label dictionary, key = filename and value = data
def read_image(file_path):
    #get the path
    path = file_path
    first_folder_name = os.listdir(path)
    first_folder_name.sort()
    label = {}
    image = {}
    for second_folder in first_folder_name:
        folder_path = path + '/' + second_folder
        file_folder_list = os.listdir(folder_path)
        file_folder_list.sort()
        for file_name in file_folder_list:
            file_path = folder_path + '/' + file_name
            file_name_list = os.listdir(file_path)
            file_name_list.sort()
            for file in file_name_list:
                image_path = file_path + '/' + file
                if file_name == "image" :
                    image[file] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)     #read image
                else :
                    label[file] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)     #read label
    return image, label


# In[4]:


print('''
# function read_image(file_path), which returns image and label in the dictionary datatype
 ''')


# In[5]:


def label_to_class_label(label):
    
    label[label == 0] = 0              
    label[label == 255] = 1
        
    return np.array(label)


# In[6]:


def label_to_1Hlabel(label):
    
    teye = np.eye(2,dtype=np.uint8)    #num_class = 2
    label_seg = np.zeros([*label.shape[:],2],dtype=np.uint8)
    label_seg [label == 0] = teye[0]
    label_seg [label == 1] = teye[1]

    return np.array(label_seg)


# In[7]:


#crop off 0~99 column and 401~last column
def crop_image(image):    
    crop_image = np.delete(np.delete(image,np.s_[451:],axis = 1), np.s_[:51], axis = 1)
    
    return crop_image


# In[8]:


#crop one image into four image(size = 300*300)
def crop_to_four(image):
    crop1_image = np.copy(image[:300])
    crop2_image = np.copy(image[300:600])
    crop3_image = np.copy(image[600:900])
    crop4_image = np.copy(image[900:1200])
    return crop1_image, crop2_image, crop3_image, crop4_image


# In[9]:


def merge(crop1, crop2, crop3, crop4):
    
    image = np.vstack([crop1, crop2, crop3, crop4])
    
    return image


# In[10]:


#normalize data by zero mean method
def zero_mean(image):
    image_mean = image.mean()
    image_max = image.max()
    image_min = image.min()
    image_norm = (image - image_mean) / (image_max - image_min)
    return image_norm


# In[11]:


#show one image and label
def imshow(image, label):
    plt.figure(figsize = (15,15))
    plt.subplot(221)
    plt.imshow(image, cmap = 'gray')
    plt.subplot(222)
    plt.imshow(label, cmap = 'gray')
    plt.show()
    return 0


# In[12]:


print('''
# function imshow(image, label), which shows two images in grayscale
''')


# In[13]:


#resize = True, then return image of size (320,320)
#to display, image[crop_displacement, slice_NO,: ,:]
def datapreprocessing(image, label, resize):
    new_image = []    
    new_label = []
        
    print("preprocessing....\n")
    print("dealing with image....\n")
    
    for x in image:
        
        crop_x = crop_image(x)
        crop1_x, crop2_x, crop3_x, crop4_x = crop_to_four(crop_x)
                
        temp_1 = cv2.equalizeHist(crop1_x)
        crop1_x = cv2.addWeighted(crop1_x ,0.5, temp_1, 0.5,0,-1)
        
        temp_2 = cv2.equalizeHist(crop2_x)
        crop2_x = cv2.addWeighted(crop2_x ,0.5, temp_2, 0.5,0,-1)
        
        temp_3 = cv2.equalizeHist(crop3_x)
        crop3_x = cv2.addWeighted(crop3_x ,0.5, temp_3, 0.5,0,-1)
        
        temp_4 = cv2.equalizeHist(crop4_x)
        crop4_x = cv2.addWeighted(crop4_x ,0.5, temp_4, 0.5,0,-1)
                       
        crop1_x = np.float32(crop1_x)
        
        crop2_x = np.float32(crop2_x)
        
        crop3_x = np.float32(crop3_x)
        
        crop4_x = np.float32(crop4_x)
        
        
        crop1_x = zero_mean(crop1_x)        
        crop2_x = zero_mean(crop2_x)
        crop3_x = zero_mean(crop3_x)
        crop4_x = zero_mean(crop4_x)
        
        x = merge(crop1_x, crop2_x, crop3_x, crop4_x)
        if (resize) :
            
            x = cv2.resize(x, (224,672))
        new_image.append(x)
        
    print("preprocessing for image finish!\n")
    print("dealing with label....\n")
  
    label = label_to_class_label(label)
    
    print("label_to_class_label finish!\n")
    
    label = label_to_1Hlabel(label)
    
    print("Function label_to_1Hlabel finish!\n")
    
    
    for y in label:
        
        y = np.float32(y)    
        
        if (resize) :
            y = cv2.resize(y, (224,672))
            
        new_label.append(y)
    print("preprocessing for label finish!\n")
    return np.array(new_image), np.array(new_label)


# In[14]:


print('''
# function datapreprocessing(image, label, resize), which finishs the preprocessing.
  Note that resize is True or False
''')


# In[15]:


def transfer_to_train_data(image, input_shape):   #input_shape is the dimension of data
    
    x = np.reshape(image,input_shape)         
    x = np.array(x)
    print("Function transfer_to_train_data finish!\n")
    return x


# In[16]:


print('''
# function transfer_to_train_data(image, input_shape), which prepares the train_x data for model  
''')


# In[17]:


#get_floder(image in dict, label in dict, image_name list, label_name list, training data size)
#return four array (train_image, train_label, test_image, test_label)

def get_floder(image, label, image_name, label_name, train_floder_size):
    
    floder_train_image = []
    floder_train_label = []
    floder_test_image = []
    floder_test_label = []

    #for image
    count1 = 0  
    for img in image_name:
        
        if count1 < train_floder_size:
            floder_train_image.append(image[img])
            count1 = count1 + 1
            
        else :
            floder_test_image.append(image[img])
            count1 = count1 + 1
    # for label

    count = 0 
    for labels in label_name:
        if count < train_floder_size:
            floder_train_label.append(label[labels])
            count = count + 1
            
        else :
            floder_test_label.append(label[labels])
            count = count + 1
            
    return np.array(floder_train_image), np.array(floder_train_label), np.array(floder_test_image) ,np.array(floder_test_label)


# In[18]:


#return three floders
def three_floder_validation(image, label):
    size = 40
    image_name = list(image.keys())
    label_name = list(label.keys())
    name_image = []
    name_label = []
    floder = [[],[],[]]
    
    floder2_image_name = image_name[:20] + image_name[40:60] + image_name[20:40]
    
    floder2_label_name = label_name[:20] + label_name[40:60] + label_name[20:40]
    
    floder3_image_name = image_name[40:60] + image_name[20:40] + image_name[:20]
    
    floder3_label_name = label_name[40:60] + label_name[20:40] + label_name[:20]
    name_image.append(image_name)
    name_image.append(floder2_image_name)
    name_image.append(floder3_image_name)
    name_label.append(label_name)
    name_label.append(floder2_label_name)
    name_label.append(floder3_label_name)
    
    
    for i in range(3):
        train_x, train_y, test_x, test_y = get_floder(image, label, name_image[i], name_label[i], 40)
        floder[i].append(train_x)
        floder[i].append(train_y)
        floder[i].append(test_x)
        floder[i].append(test_y)
        
    floder1 = floder[0]
    floder2 = floder[1]
    floder3 = floder[2]
    
    print("Function three_floder_validation finish!\n")
    return floder1, floder2, floder3


# In[19]:


print('''
# three_floder_validation(image, label), which classifies the data to three floder.
  It returns floder1, floder2, floder3
''')


# In[20]:


print('''
Sucessfully import Data_processing_pretrain_final.py!
-------------------------------------------------------------------------------
''')


# In[29]:


# read data
#image, label = read_image("data")


# In[30]:


#floder1, floder2, floder3 = three_floder_validation(image,label)


# In[31]:


#train_x, train_y, test_x, test_y = floder1[0], floder1[1], floder1[2], floder1[3]


# In[32]:


#train_x, train_y = datapreprocessing(train_x, train_y, resize = True)
#test_x, test_y = datapreprocessing(test_x, test_y, resize = True)


# In[33]:


#train_x = transfer_to_train_data(train_x, (train_x.shape[0], train_x.shape[1],train_x.shape[2],1))
#test_x = transfer_to_train_data(test_x, (test_x.shape[0], test_x.shape[1],test_x.shape[2],1))


# In[34]:


#train_y.shape


# In[35]:


#imshow(train_y[1,:,:,1],train_y[10,:,:,0])


# In[ ]:





# In[ ]:





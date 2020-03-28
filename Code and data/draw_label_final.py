#!/usr/bin/env python
# coding: utf-8

# In[19]:


print('''
----------------------------------------------------------------------------------
This draw_label_final.py

Importing functions.......
''')


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import cv2
import os


# In[9]:


color = {}
color['background'] = 0
for i in range(18):
    color['v' + str(i+1)] = (i+1)*12 + 20


# In[18]:


color_list = list(color.values())


# In[24]:


def draw_label(label):   
    color_label = []
    for image in label:
        imgray = np.copy(image)
        contours ,hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        k = 1
        for i in contours:
            cv2.fillPoly(image, [i],  (color_list[k], color_list[k], color_list[k]));
            k = k + 1
        color_label.append(image)
    
    return np.array(color_label)


# In[25]:


def image_to_class_label(image):
    image[image == color['background']] = 0              
    image[image == color['v1']] = 1
    image[image == color['v2']] = 2
    image[image == color['v3']] = 3
    image[image == color['v4']] = 4
    image[image == color['v5']] = 5
    image[image == color['v6']] = 6
    image[image == color['v7']] = 7
    image[image == color['v8']] = 8
    image[image == color['v9']] = 9
    image[image == color['v10']] = 10
    image[image == color['v11']] = 11
    image[image == color['v12']] = 12
    image[image == color['v13']] = 13
    image[image == color['v14']] = 14
    image[image == color['v15']] = 15
    image[image == color['v16']] = 16
    image[image == color['v17']] = 17
    image[image == color['v18']] = 18
    
        
    return np.array(image)


# In[26]:


def image_to_1Hlabel(label):
              
    teye = np.eye(19,dtype=np.uint8)               #num class
    
    label_seg = np.zeros([*label.shape[:],19],dtype=np.uint8)
    label_seg [label == 0] = teye[0]
    label_seg [label == 1] = teye[1]
    label_seg [label == 2] = teye[2]
    label_seg [label == 3] = teye[3]
    label_seg [label == 4] = teye[4]
    label_seg [label == 5] = teye[5]
    label_seg [label == 6] = teye[6]
    label_seg [label == 7] = teye[7]
    label_seg [label == 8] = teye[8]
    label_seg [label == 9] = teye[9]
    label_seg [label == 10] = teye[10]
    label_seg [label == 11] = teye[11]
    label_seg [label == 12] = teye[12]
    label_seg [label == 13] = teye[13]
    label_seg [label == 14] = teye[14]
    label_seg [label == 15] = teye[15]
    label_seg [label == 16] = teye[16]
    label_seg [label == 17] = teye[17]
    label_seg [label == 18] = teye[18]
   
    
    
    return label_seg


# In[20]:


print('''
Sucessfully import draw_label_final.py!
-------------------------------------------------------------------------------
''')


# In[ ]:





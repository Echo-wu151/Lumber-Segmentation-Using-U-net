#!/usr/bin/env python
# coding: utf-8

# In[2]:


print('''
----------------------------------------------------------------------------------
This file is semantic_Unet_pretrain.py

Importing model.......
''')


# In[10]:


import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, concatenate, BatchNormalization


# In[2]:


class History(tf.keras.callbacks.Callback):
    
    def on_train_begin(self, logs):
        self.losses = []
        self.validation_loss = []
        self.accuracy = []
        self.validation_accuracy = []
    
    def on_batch_end(self, batch, logs):
        self.losses.append(logs.get('loss'))
        self.validation_loss.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('acc'))
        self.validation_accuracy.append(logs.get('val_acc'))


# In[13]:

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = "best_pretrain_weights.h5",
                                                      save_best_only = True ,
                                                      monitor = 'val_loss',
                                                      verbose = 1)

# In[23]:


print(
'''

# use function checkpoint(save_name) for saving best weighting.
# use class History() for creating a object which saves loss and accuracy.
# use callbacks = [checkpoint, history]

''')


# In[5]:


def dice_coef(y_true, y_pred, smooth= 1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


# In[6]:


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# In[7]:


class build_model():
            
    def __init__ (self, filters, kernel, activation, padding, NO):
        self.filters = filters
        self.kernel = kernel
        self.activation = activation
        self.padding = padding
        self.NO = NO
    
    def contraction(self, inputs):
        self.conv1 = Conv2D(filters = self.filters, kernel_size = self.kernel, activation = self.activation, 
                            padding = self.padding, name = "c_" + str(self.NO) + "_conv1")(inputs)
        self.BN1 = BatchNormalization(name = "c_" + str(self.NO) + "_BN1")(self.conv1)
        self.conv2 = Conv2D(filters = self.filters, kernel_size = self.kernel, activation = self.activation, 
                            padding = self.padding, name = "c_" + str(self.NO) + "_conv2")(self.BN1)
        self.BN2 = BatchNormalization(name = "c_" + str(self.NO) + "_BN2")(self.conv2)
        self.maxpooling = MaxPooling2D(pool_size = (2, 2), strides = (2,2), name = "c_" + str(self.NO) + "_maxpool1")(self.BN2)
        
    def bottom(self, inputs):
        self.conv1 = Conv2D(filters = self.filters, kernel_size = self.kernel, activation = self.activation, 
                            padding = self.padding, name = "b_conv1")(inputs)
        self.BN1 = BatchNormalization(name = "b_BN1")(self.conv1)
        self.conv2 = Conv2D(filters = self.filters, kernel_size = self.kernel, activation = self.activation, 
                            padding = self.padding, name = "b_conv2")(self.BN1)
        self.BN2 = BatchNormalization(name = "b_BN2")(self.conv2)
        
    def expansion(self, inputs, concatenatae_layer):
        self.T_conv = Conv2DTranspose(filters = self.filters, kernel_size = self.kernel, activation = self.activation, 
                            padding = self.padding, strides = (2,2), name = "e_" + str(self.NO) + "_Tconv")(inputs)
        self.up_cov = concatenate([concatenatae_layer, self.T_conv], axis = -1)
        self.conv1 = Conv2D(filters = self.filters, kernel_size = self.kernel, activation = self.activation, 
                            padding = self.padding, name = "e_" + str(self.NO) + "_conv1")(self.up_cov)
        self.BN1 = BatchNormalization(name = "e_" + str(self.NO) + "_BN1")(self.conv1)
        self.conv2 = Conv2D(filters = self.filters, kernel_size = self.kernel, activation = self.activation, 
                            padding = self.padding, name = "e_" + str(self.NO) + "_conv2")(self.BN1)
        self.BN2 = BatchNormalization(name = "e_" + str(self.NO) + "_BN2")(self.conv2)


# In[8]:


def u_net(input_shape):
    
    #intialize some layers
    kernel_size = (9,9)
    filter_size = 16
    padding = "same"
    lr = 0.0001
    num_class = 2
    
    contraction1 = build_model(filters = filter_size, kernel = kernel_size, activation = "relu",
                               padding = padding, NO = 1)
    contraction2 = build_model(filters = filter_size * 2, kernel = kernel_size, activation = "relu",
                               padding = padding, NO =2)
    contraction3 = build_model(filters = filter_size * 4, kernel = kernel_size, activation = "relu",
                               padding = padding, NO =3)
    contraction4 = build_model(filters = filter_size * 8, kernel = kernel_size, activation = "relu",
                               padding = padding, NO =4)
    bottom = build_model(filters = filter_size * 16, kernel = kernel_size, activation = "relu",
                         padding = padding, NO = None)
    expansion1 = build_model(filters = filter_size * 8, kernel = kernel_size, activation = "relu",
                             padding = padding, NO = 1)
    expansion2 = build_model(filters = filter_size * 4, kernel = kernel_size, activation = "relu",
                             padding = padding, NO = 2)
    expansion3 = build_model(filters = filter_size * 2, kernel = kernel_size, activation = "relu",
                             padding = padding, NO = 3)
    expansion4 = build_model(filters = filter_size , kernel = kernel_size, activation = "relu",
                             padding = padding, NO = 4)
        
    #connect all layers
    
    inputs = Input(shape = input_shape, name = "input")
    contraction1.contraction(inputs)
    contraction2.contraction(contraction1.maxpooling)
    contraction3.contraction(contraction2.maxpooling)
    contraction4.contraction(contraction3.maxpooling)
    bottom.bottom(contraction4.maxpooling)
    expansion1.expansion(bottom.BN2, contraction4.BN2)
    expansion2.expansion(expansion1.BN2, contraction3.BN2)
    expansion3.expansion(expansion2.BN2, contraction2.BN2)
    expansion4.expansion(expansion3.BN2, contraction1.BN2)

    out_layer = Conv2D(filters = num_class, kernel_size = (1,1),activation = 'softmax' , padding = "valid")(expansion4.BN2)
    
    model = Model(inputs, out_layer)
    
    opt = tf.keras.optimizers.Adam(lr = lr)    
    model.compile(optimizer= opt, loss = dice_coef_loss, metrics = [dice_coef])
    
    print(model.summary())
    return model


# In[1]:


print('''
# use model = u_net(input_shape) to create a mode.

Sucessfully import semantic_Unet_pretrain.py!

-------------------------------------------------------------------------------
''')


# In[ ]:





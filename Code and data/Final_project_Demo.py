#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Data_processing_for_multi_class_final import *
from semantic_Unet import *


# In[2]:


model_floder1 = u_net((672,224,1))
model_floder2 = u_net((672,224,1))
model_floder3 = u_net((672,224,1))


# In[3]:


model_floder1.load_weights("weights/best_weights_f1.h5")
model_floder2.load_weights("weights/best_weights_f2.h5")
model_floder3.load_weights("weights/best_weights_f3.h5")


# In[4]:


def class_label_to_color_image(label):
    
    label_seg = np.zeros([*label.shape[:],3],dtype=np.uint8)
    label_seg [label == 0] = np.array([0, 0, 0])
    label_seg [label == 1] = np.array([0, 0, 255])
    label_seg [label == 2] = np.array([0, 255, 0])
    label_seg [label == 3] = np.array([255, 0, 0])
    label_seg [label == 4] = np.array([0, 255, 255])
    label_seg [label == 5] = np.array([255, 0, 255])
    label_seg [label == 6] = np.array([0, 255, 128])
    label_seg [label == 7] = np.array([128, 0, 255])
    label_seg [label == 8] = np.array([0, 128, 255])
    label_seg [label == 9] = np.array([255, 0, 128])
    label_seg [label == 10] = np.array([255, 128, 0])
    label_seg [label == 11] = np.array([0, 255, 128])
    label_seg [label == 12] = np.array([128, 255, 0])
    label_seg [label == 13] = np.array([128, 128, 255])
    label_seg [label == 14] = np.array([128, 255, 128])
    label_seg [label == 15] = np.array([255, 128, 128])
    label_seg [label == 16] = np.array([64, 128, 255])
    label_seg [label == 17] = np.array([255, 128, 64])
    label_seg [label == 18] = np.array([128, 64, 255])
    
       
    return np.array(label_seg, dtype = np.uint8)


# In[5]:


def pred_image_to_binary(label, vertebrae):
    
    label_seg = np.zeros([*label.shape[:]],dtype=np.uint8)
    for i in range(1,19):
        if i == vertebrae:
            label_seg [label == i] = 1
        else:
            label_seg [label == i] = 0
       
    return np.array(label_seg, dtype = np.uint8)


# In[6]:


def DC(pred, label):
    smooth = 1
    pred_f = pred.flatten()
    label_f = label.flatten()
    if sum(label_f) == 0:
        return -1
    intersection = 0
    for i in range(len(pred_f)):
        product = pred_f[i]*label_f[i]
        intersection = intersection + product
    dc = (2* intersection + smooth) / (sum(pred_f) + sum(label_f) + smooth)
    return round(dc, 3)
    


# In[7]:


import sys
import os
import cv2
from PyQt5.QtWidgets import *

from PyQt5 import QtCore, QtGui, QtWidgets


class MainForm(QWidget):
    def __init__(self, name = 'MainForm'):
        super(MainForm,self).__init__()
        self.setWindowTitle(name)
        self.cwd = os.getcwd() 
        self.resize(1500, 900)   
        
        
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(30, 130, 224, 672))
        self.label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(""))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(350, 130, 224, 672))
        self.label_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(""))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.setGeometry(QtCore.QRect(670, 130, 224, 672))
        self.label_3.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap(""))
        self.label_3.setObjectName("label_3")
        self.label_12 = QtWidgets.QLabel(self)
        self.label_12.setGeometry(QtCore.QRect(990, 130, 224, 672))
        self.label_12.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_12.setText("")
        self.label_12.setPixmap(QtGui.QPixmap(""))
        self.label_12.setObjectName("label_12")
        
       
        self.PushButton_chooseFile = QPushButton(self)
        self.PushButton_chooseFile.setGeometry(QtCore.QRect(350, 60, 191, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.PushButton_chooseFile.setFont(font)
        self.PushButton_chooseFile.setObjectName("PushButton_chooseFile")  
        self.PushButton_chooseFile.setText("select image")
        
        
        info2 = ["model_folder" + str(i) for i in range(1,4)]
        self.comboBox_2 = QtWidgets.QComboBox(self)
        self.comboBox_2.setGeometry(QtCore.QRect(110, 60, 191, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.comboBox_2.setFont(font)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItems(info2)
        
        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setGeometry(QtCore.QRect(120, 10, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self)
        self.label_5.setGeometry(QtCore.QRect(370, 10, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        
        self.label_13 = QtWidgets.QLabel(self)
        self.label_13.setGeometry(QtCore.QRect(620, 10, 300, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")

              
        self.label_6 = QtWidgets.QLabel(self)
        self.label_6.setGeometry(QtCore.QRect(120, 850, 81, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        
        self.label_7 = QtWidgets.QLabel(self)
        self.label_7.setGeometry(QtCore.QRect(400, 850, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self)
        self.label_8.setGeometry(QtCore.QRect(675, 850, 300, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self)
        self.label_9.setGeometry(QtCore.QRect(1300, 80, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self)
        self.label_10.setGeometry(QtCore.QRect(1300, 80, 171, 700))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        
        self.label_11 = QtWidgets.QLabel(self)
        self.label_11.setGeometry(QtCore.QRect(1070, 850, 81, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
                
        layout = QVBoxLayout()
        
        layout.addWidget(self.PushButton_chooseFile)
        
        self.PushButton_chooseFile.clicked.connect(self.chooseFile)
        
                
        _translate = QtCore.QCoreApplication.translate
        self.label_4.setText(_translate("self", "Select Image"))
        self.label_5.setText(_translate("self", "Select Model"))
        
        self.label_6.setText(_translate("self", "Source"))
        self.label_7.setText(_translate("self", "Ground truth"))
        self.label_8.setText(_translate("self", "Color Ground truth"))
        self.label_9.setText(_translate("self", "DC :"))
        
        self.label_11.setText(_translate("self", "Result"))
        

    def chooseFile(self):
        _translate = QtCore.QCoreApplication.translate
        self.label_10.setText(_translate("self", ""))
        self.label_13.setText(_translate("self", ""))
        selected_model = self.comboBox_2.currentText()
        #choose image
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,  
                                    "select image",  
                                    self.cwd, 
                                    "All Files (*);;Text Files (*.txt)")
        if fileName_choose == "":
            print("Please select an image.\n")
            return 0
        label_path = ""
        label_path_list = fileName_choose.split('/')
        for i in range(len(label_path_list)):
            if label_path_list[i] == 'image':
                text_ = "This is image " + str(label_path_list[i+1])
                self.label_13.setText(_translate("self", text_))
                label_path_list[i] = 'label'
            label_path = label_path +'/' + label_path_list[i]
        #read image and processing
        img = cv2.imread(fileName_choose, cv2.IMREAD_GRAYSCALE)        
        label = cv2.imread(label_path[1:], cv2.IMREAD_GRAYSCALE)
        #image for display
        img_show = crop_image(img)
        img_show = cv2.resize(img_show, (224, 672))        
        label_show = crop_image(label)
        label_show = cv2.resize(label, (224, 672))
        
        height, width = img_show.shape
        bytesPerLine = width
        #image for prediction
        img ,label = datapreprocessing([img], [label], True)
        img = transfer_to_train_data(img, (img.shape[0], img.shape[1], img.shape[2], 1))
        #model selection
                        
        #show image
        QImg = QtGui.QImage(img_show.data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
        QLable = QtGui.QImage(label_show.data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
        pixmap_img = QtGui.QPixmap.fromImage(QImg)
        pixmap_label = QtGui.QPixmap.fromImage(QLable)
        self.label.setPixmap(QtGui.QPixmap(pixmap_img))
        self.label_2.setPixmap(QtGui.QPixmap(pixmap_label))
        
        
        if selected_model == "model_floder1":
            pred = model_floder1.predict(img)
        elif selected_model == "model_floder2":
            pred = model_floder2.predict(img)
        else:
            pred = model_floder3.predict(img)
        
        #result processing
        pred = np.argmax(pred[0], axis = -1)
        pred = np.uint8(pred)
        
        dice_list = []
        for v in range(label.shape[3] - 1):
            temp = pred_image_to_binary(pred, v + 1)
            dice = DC(temp, label[:,:,:,v + 1])
            dice_list.append(dice)
        
        pred = class_label_to_color_image(pred)
        pred_show = cv2.cvtColor(img_show, cv2.COLOR_GRAY2RGB)
        label_show = cv2.cvtColor(label_show, cv2.COLOR_GRAY2RGB)
        
        color_label = np.copy(label[0])
        
        color_label = np.argmax(color_label, axis = -1)
        
        color_label = class_label_to_color_image(color_label)
        color_label = np.uint8(color_label)
        
        added_image_2 = cv2.addWeighted(pred_show,0.75 ,pred,0.25, 0)
            
        #show image
        Qcolor = QtGui.QImage(color_label.data, width, height, bytesPerLine * 3, QtGui.QImage.Format_RGB888)        
        pixmap_color_label = QtGui.QPixmap.fromImage(Qcolor)                
        self.label_3.setPixmap(QtGui.QPixmap(pixmap_color_label))

        Qpred = QtGui.QImage(added_image_2.data, width, height, bytesPerLine * 3, QtGui.QImage.Format_RGB888)        
        pixmap_pred_2 = QtGui.QPixmap.fromImage(Qpred)                
        self.label_12.setPixmap(QtGui.QPixmap(pixmap_pred_2))
        text = ""
        j = 1
        s = 0
        for i in range(len(dice_list)-1,-1,-1):
            if dice_list[i] != -1:
                text = text + "v" + str(j) + "：" + str(dice_list[i]) + "\n"
                j = j + 1
                s = s + dice_list[i]
                
        text = text + "Avg : " + str(round(s/(j - 1),3))
        
        self.label_10.setText(_translate("self", text))
        print("done!\n ----------------------------------------\n")
        
        
    
   
import sys
app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
mainForm = MainForm('Final_project')
mainForm.show()
app.exec_()



# In[ ]:





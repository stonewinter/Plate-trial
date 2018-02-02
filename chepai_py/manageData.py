import cv2
import opencvlib
import numpy as np
import tensorflow as tf
import os
from glob import glob

"""
本程序用于 将T/F_candidate下整理好的图片进行像素特征提取，进一步构建feature,label数据表
保存在  features.npy  labels.npy
"""



##################
# True-> [1 0] 是车牌的话就是[1 0]
##################
names = glob("D:\Python saved program\OpenCV_basic1\chepai\T_candidate\*.jpg");
X = [];
for idx,name in enumerate(names):
    img = cv2.imread(name);
    # _,img = opencvlib.GetOtusImage(img);
    resize = opencvlib.Resize(img,60,20);
    X.append(resize.flatten()/255);
featureSet_1 = np.array(X)
labelSet_1 = np.zeros((featureSet_1.shape[0],2),dtype=np.float32);
labelSet_1[:,0] = 1;
# print("featureSet_1",featureSet_1.shape)
# print(featureSet_1)
# print("labelSet_1",labelSet_1.shape)
# print(labelSet_1)



##################
# False-> [0 1]
##################
names = glob("D:\Python saved program\OpenCV_basic1\chepai\F_candidate\*.jpg");
X = [];
for idx,name in enumerate(names):
    img = cv2.imread(name);
    # _, img = opencvlib.GetOtusImage(img);
    resize = opencvlib.Resize(img,60,20);
    X.append(resize.flatten()/255);
featureSet_2 = np.array(X)
labelSet_2 = np.zeros((featureSet_2.shape[0],2),dtype=np.float32);
labelSet_2[:,1] = 1;
# print("featureSet_2",featureSet_2.shape)
# print(featureSet_2)
# print("labelSet_2",labelSet_2.shape)
# print(labelSet_2)



featureSet = np.vstack((featureSet_1,featureSet_2));
labelSet = np.vstack((labelSet_1,labelSet_2) );
print("featureSet",featureSet.shape)
print(featureSet)
print("labelSet",labelSet.shape)
print(labelSet)


np.save("./features.npy",featureSet);
np.save("./labels.npy",labelSet);




# featureSet = np.load("features.npy")
# labelSet = np.load("labels.npy")
# print("featureSet",featureSet.shape)
# print(featureSet)
# print("labelSet",labelSet.shape)
# print(labelSet)
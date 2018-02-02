import cv2;
import numpy as np;
import opencvlib;
import math;
import random;
import os
import glob

"""""""""""""""""""""""""""""""""""""""""""""
本程序用于将chepai/T_candidate/下的各个图片统一改名为 "数字.jpg"
"""""""""""""""""""""""""""""""""""""""""""""

def GetFileNumInDir(filePath):
    return len(os.listdir(filePath));


names = glob.glob("../chepai/T_candidate/*.jpg")
for idx,name in enumerate(names):
    img = cv2.imread(name);
    # cv2.imshow(str(idx),img)
    cv2.imwrite("../chepai/T_candidate/"+str(idx)+".jpg", img);
    os.remove(name);


names = glob.glob("../chepai/F_candidate/*.jpg")
for idx,name in enumerate(names):
    img = cv2.imread(name);
    # cv2.imshow(str(idx),img)
    cv2.imwrite("../chepai/F_candidate/"+str(idx)+".jpg", img);
    os.remove(name);


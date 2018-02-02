import cv2
import numpy as np
import opencvlib

def nothing(x):
    pass


img = cv2.imread("../chepai/4.jpg")
img = opencvlib.Resize(img,500,450)
# img = opencvlib.GetGrayImage(img)
b,g,r = cv2.split(img);
cv2.namedWindow('image')
cv2.createTrackbar('maxVariation','image',0,15,nothing)


h,w = img.shape[0:2];
imgArea = h*w
minAreaThresh = int(imgArea*0.0005);
maxAreaThresh = int(imgArea*0.01);


MSER = opencvlib.GetBlankImg(w, h);
while(1):
    cv2.imshow("MSER", MSER);
    k=cv2.waitKey(1)&0xFF
    if k==27:
        break

    at=cv2.getTrackbarPos('maxVariation','image')
    MSER = opencvlib.GetBlankImg(w, h);

    regions = opencvlib.MSER(b
                             , minArea2del=minAreaThresh
                             , maxArea2del=maxAreaThresh
                             , minDiversity=0.5
                             , maxVariation=0.12);
    for reg in regions:
        for point in reg:
            row = point[1];
            col = point[0];
            if (MSER[row, col] == 0):
                MSER[row, col] = 255;


cv2.destroyAllWindows()
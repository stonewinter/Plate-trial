import cv2;
import numpy as np;
import opencvlib;
import tensorflow as tf;

"""""""""""""""""""""""""""""""""""""""
本程序用于 定位一张图片中的车牌，并将其显示出来
"""""""""""""""""""""""""""""""""""""""





plate_w_h_ratio = 25/70;

def findPlates(
         img,
         minAreaRatio2del = 0.004,
         maxAreaRatio2del = 0.75,
         resize_w= 300,
         resize_h = int(200*plate_w_h_ratio),
         veri_edges_min = 35,
         hori_edges_max = 70
        ):
    ############## extract blue region ##############
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
    lower_blue = np.array([100,43,46]);
    upper_blue = np.array([124,255,255]);
    mask = cv2.inRange(hsvImg, lower_blue, upper_blue); # set the pixel which has the color value within the range to true, otherwise to false.
    res = cv2.bitwise_and(hsvImg, hsvImg, mask = mask);
    # img2 = cv2.cvtColor(res, cv2.COLOR_HSV2BGR);

    ############### get higt-contrast gray image ##############
    img2 = opencvlib.GetGrayImage(res)
    # cv2.imshow("gray", img2);
    img2 = opencvlib.HistGlobalEqualize(img2)
    # cv2.imshow("gray Hist", img2);


    ############### otus and filter noise ##############
    thresh = opencvlib.GrayGTthresh2White(img2,130);
    # _, thresh = opencvlib.GetOtusImage(img2)
    # cv2.imshow("thresh", thresh);

    horiz_kernel = opencvlib.GetAreaKernel(cv2.MORPH_RECT,1,3);
    veri_kernel = opencvlib.GetAreaKernel(cv2.MORPH_RECT,3,1);
    hori_left_kernel = np.array([ [1,1,1] ]);
    # veri_down_kernel = np.array([ [1,1,1] ])[0][:,np.newaxis];
    squre_kernel = opencvlib.GetAreaKernel(cv2.MORPH_RECT,3,3);

    H_erosion = cv2.erode(thresh,horiz_kernel,iterations=1);
    H_dilation = cv2.dilate(H_erosion,horiz_kernel,iterations=1);
    # cv2.imshow("H--fliter", H_dilation);

    V_erosion = cv2.erode(H_dilation,veri_kernel,iterations=1);
    V_dilation = cv2.dilate(V_erosion,veri_kernel,iterations=1);
    # cv2.imshow("V--filter", V_dilation);

    H_blob = cv2.dilate(V_dilation,horiz_kernel,iterations=15);
    H_blob = cv2.erode(H_blob,veri_kernel,iterations=1);
    H_blob = cv2.dilate(H_blob,veri_kernel,iterations=1);
    # H_blob = cv2.dilate(H_blob, hori_left_kernel, iterations=6, anchor=(0, 0));
    # H_blob = cv2.dilate(H_blob, veri_down_kernel, iterations=5, anchor=(0, 0));
    # cv2.imshow("H--blob", H_blob);




    # find closed regions
    contourList = opencvlib.GetOuterContours(H_blob,100)
    # print("contourlist\n",contourList)
    # print("how many contours?",len(contourList))
    # colorImg = cv2.cvtColor(H_blob,cv2.COLOR_GRAY2BGR);
    # originalImg,contoursImg = opencvlib.DrawAllContoursOnImage(colorImg,contourList);
    # cv2.imshow("contour",contoursImg)


    # find contour regions whose area: *** <= area <= ***
    # minAreaRatio2del = 0.015
    # maxAreaRatio2del = 0.75
    h,w = img.shape[0:2]
    imgArea = h*w;
    idxes = [];
    for idx,c in enumerate(contourList):
        area = opencvlib.GetContourArea(c);
        if (area<=imgArea*minAreaRatio2del
            or area>=imgArea*maxAreaRatio2del):
            pass
        else:
            idxes.append(idx);

    newContourList = [];
    for idx in idxes:
        newContourList.append(contourList[idx]);
    contourList = newContourList;
    del newContourList;



    # blankImg = opencvlib.GetBlankImg(w, h);
    # cv2.drawContours(blankImg, contourList, -1, 255, -1);
    # cv2.imshow("area LEFT", blankImg);


    ##########################
    ####  interest roi    ####
    ##########################
    interest_roi_list = [];

    # resize_w = 350;
    # resize_h = 200;
    # veri_edges_min = 20
    # hori_edges_max = 80
    for idx,c in enumerate(contourList):
        rectContourPoints = opencvlib.FindOptRectContour(c);
        blankImg = opencvlib.GetBlankImg(w, h);
        cv2.drawContours(blankImg, rectContourPoints, -1, 255, -1);

        x = np.where(blankImg>0)[::-1][0]
        y = np.where(blankImg>0)[::-1][1]
        roi_x1 = np.min(x)
        roi_x2 = np.max(x)
        roi_y1 = np.min(y)
        roi_y2 = np.max(y)
        interest_roi = img[roi_y1:roi_y2, roi_x1:roi_x2];
        interest_roi = opencvlib.Resize(interest_roi,resize_w,resize_h);


        # 进行横向edge统计,数量多的interest roi被舍弃
        GaussianBlur_roi = cv2.GaussianBlur(interest_roi, (7, 7), 0);
        sobely_roi = cv2.Sobel(GaussianBlur_roi, cv2.CV_8U, 0, 1, ksize=3);
        canny_horizontal = cv2.Canny(sobely_roi, 50, 100, apertureSize=3, L2gradient=True);
        # cv2.imshow("canny_vertical" + str(idx), canny_vertical);
        colList = np.where(canny_horizontal>0)[::-1][0].tolist();
        if(len(colList) > 0): # 如果存在横向edge的话
            colSet = set(colList);
            tempSet = set();
            for col in colSet:
                tempSet.add(colList.count(col))
            maxHoriCount = max(list(tempSet));
            print(" maxHoriCount"+str(idx),maxHoriCount)
            if(maxHoriCount>=hori_edges_max): # 如果存在横向edge太多的话
                # cv2.imshow("canny_horizontal filter out" + str(idx)+" maxHoriCount"+str(maxHoriCount), canny_horizontal);
                interest_roi = None;
        else: # 如果不存在横向edge的话
            pass;



        # 进行纵向edge统计,数量多的就是车牌的那个interest roi
        if(interest_roi is not None):
            sobelx_roi = cv2.Sobel(GaussianBlur_roi, cv2.CV_8U, 1, 0, ksize=3);
            canny_vertical = cv2.Canny(sobelx_roi, 50, 100, apertureSize=3, L2gradient=True);
            # cv2.imshow("canny_vertical" + str(idx), canny_vertical);
            rowList = np.where(canny_vertical > 0)[::-1][1].tolist();
            if (len(rowList) > 0): # 如果存在纵向edge的话
                rowSet = set(rowList);
                tempSet = set();
                for row in rowSet:
                    tempSet.add(rowList.count(row))
                maxVertCount = max(list(tempSet));
                print(" maxVertCount" + str(idx), maxVertCount)
                if (maxVertCount < veri_edges_min): # 如果纵向edge太少的话
                    # cv2.imshow("canny_vertical filter out" + str(idx) + " maxVertCount" + str(maxVertCount), canny_vertical);
                    interest_roi = None;
            else: # 如果不存在纵向edge的话
                interest_roi = None;


        if (interest_roi is not None):
            # cv2.imshow("region"+str(idx),interest_roi);
            # bestThreshold, otusImg = opencvlib.GetOtusImage(interest_roi)
            # cv2.imshow("otus img"+str(idx),otusImg)
            interest_roi_list.append(interest_roi);

    return interest_roi_list;








def PerspectiveAdjust(img):
    thresh, otusImg = opencvlib.GetOtusImage(img);
    cv2.imshow("otus", otusImg)












# img = cv2.imread("./chepai_test/chepai_test7.jpg");
img = cv2.imread("../chepai/28.jpg");





print("************ start looking for plate region candidates *************")
interest_roi_list = findPlates(img);
if(interest_roi_list is not None):
    tempList = [];
    for eachROI in interest_roi_list:
        roi = findPlates(eachROI,
                         minAreaRatio2del=0.05,
                         maxAreaRatio2del=0.9,
                         veri_edges_min=35,
                         hori_edges_max=70);
        if (len(roi) > 0):
            roi = roi[0]
            tempList.append(roi);

    interest_roi_list = tempList;
    del tempList;
else:
    interest_roi_list = [];



print("************ building up plate recognition net *************")
featureSet = np.load("features.npy");
labelSet = np.load("labels.npy");
W = tf.Variable(tf.zeros([featureSet.shape[1], labelSet.shape[1]]), dtype=tf.float32)
b = tf.Variable(tf.zeros([1, labelSet.shape[1]]), dtype=tf.float32)
sess = tf.Session();
tf.train.Saver().restore(sess, "./PlateRecogNet/PlateRecogNet.ckpt")
Xholder = tf.placeholder(tf.float32, [None, featureSet.shape[1]])
Yholder = tf.placeholder(tf.float32, [None, labelSet.shape[1]])
# net architecture
predict_outputs = tf.nn.softmax(tf.matmul(Xholder, W) + b);





print("************ start verifying real plate among candidates *************")
if(len(interest_roi_list)>1):
    for idx,interest_roi in enumerate(interest_roi_list):
        cv2.imshow("region" + str(idx), interest_roi);
        print("shape" + str(idx), interest_roi.shape)
        resize = opencvlib.Resize(interest_roi, 60, 20);
        X = np.array([resize.flatten() / 255]);
        prediction = sess.run(predict_outputs, feed_dict={Xholder: X});
        result = np.argmax(prediction, 1);
        print("result" + str(idx),result)
        if(result[0] == 0):
            # use NN to test if it's a plate
            plateImg = interest_roi;
            cv2.imshow("plate" + str(idx), plateImg);

elif(len(interest_roi_list)==1):
    plateImg = interest_roi_list[0];
else:
    plateImg = None;
    print("can not find plates");





# if(plateImg is not None):
#     cv2.imshow("The plate", plateImg);
#     PerspectiveAdjust(plateImg)







print("DONE");




opencvlib.WaitEscToExit();
cv2.destroyAllWindows();

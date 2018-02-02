import cv2;
import numpy as np;
import opencvlib;
import tensorflow as tf;
from matplotlib import pyplot as plt

"""""""""""""""""""""""""""""""""""""""
本程序用于 定位一张图片中的车牌，并将其显示出来
"""""""""""""""""""""""""""""""""""""""

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
print("************ done with building up plate recognition net *************")

def IsPlate(img):
    resize = opencvlib.Resize(img, 60, 20);
    X = np.array([resize.flatten() / 255]);
    prediction = sess.run(predict_outputs, feed_dict={Xholder: X});
    result = np.argmax(prediction, 1);
    # print("result" + str(idx),result)
    if (result[0] == 0):
        # use NN to test if it's a plate
        return True;
    else:
        return False;



def CleanMask(mask,minArea,maxArea):
    return opencvlib.GetContourRegionMask(mask,
                opencvlib.CleanContoursByArea(opencvlib.GetOuterContours(mask, 100),
                minArea, maxArea));



plate_w_h_ratio = 25/70;

def findPlates(
         img,
         minAreaRatio2del = 0.004,
         maxAreaRatio2del = 0.75,
         resize_w= 200,
         resize_h = int(200*plate_w_h_ratio),
         veri_edges_min = 35,
         hori_edges_max = 70
        ):
    ############## extract blue region ##############
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
    lower_blue = np.array([100,44,48]);
    upper_blue = np.array([133,255,255]);
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
    roi_coord_list = [];

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
            roiCoords = [roi_x1,roi_x2,roi_y1,roi_y2]
            roi_coord_list.append(roiCoords)

    return interest_roi_list, roi_coord_list;




# img = cv2.imread("./chepai_test/chepai_test6.jpg");
img = cv2.imread("../chepai/1.jpg");


def PerspectivePlate(plateImg):
    plate = plateImg.copy();
    cv2.imshow("ori plate",plateImg)

    # # 过滤蓝色
    # hsvImg = cv2.cvtColor(plateImg, cv2.COLOR_BGR2HSV);
    # lower_blue = np.array([100, 44, 48]);
    # upper_blue = np.array([133, 255, 255]);
    # mask = cv2.inRange(hsvImg, lower_blue, upper_blue);
    # tmp = cv2.bitwise_and(plateImg, plateImg, mask=mask);
    # tmp = cv2.cvtColor(tmp, cv2.COLOR_HSV2BGR)
    # tmp = opencvlib.HistGlobalEqualize(tmp)

    # blur = cv2.GaussianBlur(tmp, (7, 7), 0);
    # blueMask = opencvlib.GetGrayImage(blur)
    # blueMask = opencvlib.GetOtusImage(blur)[1]
    # plateImg = cv2.bitwise_and(plateImg, plateImg, mask=blueMask);

    b, g, r = cv2.split(plateImg);
    otus_mask = opencvlib.GetOtusImage(plateImg)[1]
    otus_mask = CleanMask(otus_mask, 20, otus_mask.shape[0]*otus_mask.shape[1])
    cv2.imshow("otus_mask", otus_mask)
    # otus_and = cv2.bitwise_and(plateImg,plateImg,mask=otus_mask);
    # cv2.imshow("otus_and", otus_and)


    # red_mask = opencvlib.GrayGTthresh2White(opencvlib.HistGlobalEqualize(r), 100)
    # cv2.imshow("red_mask", red_mask)
    # blue_mask = opencvlib.GrayGTthresh2White(b,130)
    blue_mask = opencvlib.GetOtusImage(b)[1]
    blue_mask = CleanMask(blue_mask,20, otus_mask.shape[0]*otus_mask.shape[1])
    cv2.imshow("blue_mask", blue_mask)



    b_otus_and = cv2.bitwise_and(blue_mask,otus_mask)
    b_otus_and = CleanMask(b_otus_and, 20, otus_mask.shape[0]*otus_mask.shape[1])
    # b_r_and = cv2.bitwise_and(blue_mask, red_mask)

    cv2.imshow("trial mask", b_otus_and)
    sobelx_roi = cv2.Sobel(b_otus_and, cv2.CV_64F, 1, 0, ksize=1);
    sobelx_roi = np.absolute(sobelx_roi)
    # cv2.imshow("edges sobel", sobelx_roi)
    # count = []
    # for eachRow in range(sobelx_roi.shape[0]):
    #     count.append(opencvlib.HorizontalPixelCount(sobelx_roi,eachRow))
    # print( max(count) )




    # blue_left_out = cv2.bitwise_and(plateImg, plateImg, mask=blue_mask);
    # cv2.imshow("blue_left_out", blue_left_out)
    #
    # blue_gray = opencvlib.GetGrayImage(blue_left_out)
    # cv2.imshow("blue_gray",blue_gray)

    # ot = cv2.bitwise_not(opencvlib.GetOtusImage(plateImg)[1])
    #
    # ot = cv2.bitwise_and(ot, mask);
    #
    # ot = opencvlib.GetContourRegionMask(ot,
    #                                       opencvlib.CleanContours(ot, opencvlib.GetOuterContours(ot, 100),
    #                                       0.02, 0.98))
    # cv2.imshow("ot ot", ot)
    #
    # sobelx_roi = cv2.Sobel(ot, cv2.CV_8U, 1, 0, ksize=1);
    # # sobely_roi = cv2.Sobel(ot, cv2.CV_8U, 0, 1, ksize=3);
    # # sobel_and = cv2.bitwise_and(sobelx_roi, sobely_roi)
    # cv2.imshow("edges sobel and", sobelx_roi)
    # canny_x = cv2.Canny(sobelx_roi, 100, 150, apertureSize=3, L2gradient=True);
    # cv2.imshow("edges", canny_x)
    #
    #


    # horiz_kernel = opencvlib.GetAreaKernel(cv2.MORPH_RECT, 1, 5);
    # veri_kernel = opencvlib.GetAreaKernel(cv2.MORPH_RECT, 3, 1);
    # h, w = blue_mask.shape[0:2]
    # blankImg = opencvlib.GetBlankImg(w, h)
    # blankImg[:, 0:w // 2] = blue_mask[:, 0:w // 2]
    # left = cv2.dilate(blankImg, horiz_kernel, iterations=3, anchor=(4, 0));
    # blankImg[:, :] = 0
    # blankImg[:, w // 2:w] = blue_mask[:, w // 2:w]
    # right = cv2.dilate(blankImg, horiz_kernel, iterations=3, anchor=(0, 0));
    # blankImg[:, :] = 0
    # blankImg[0:h // 2, :] = blue_mask[0:h // 2, :]
    # top = cv2.dilate(blankImg, veri_kernel, iterations=2, anchor=(0, 2));
    # blankImg[:, :] = 0
    # blankImg[h // 2:h, :] = blue_mask[h // 2:h, :]
    # bottom = cv2.dilate(blankImg, veri_kernel, iterations=2, anchor=(0, 0));
    # mask = cv2.bitwise_or(left, right);
    # mask = cv2.bitwise_or(mask, top);
    # mask = cv2.bitwise_or(mask, bottom);
    # cv2.imshow("expand mask", mask)

    caliAngle = opencvlib.RotatedImgCalibrate(sobelx_roi);
    rotatePlate = opencvlib.Rotate(plate, caliAngle)
    rotateMask = opencvlib.Rotate(sobelx_roi, caliAngle)
    cv2.imshow("rotateMask", rotateMask)
    vertices = opencvlib.FindRotatedImgClimaxes(rotateMask)
    topleft, topright, botright, botleft = vertices
    h, w = rotatePlate.shape[0:2]
    plate = opencvlib.PerspectiveTransform(rotatePlate,
                                           [topleft, topright, botright, botleft],
                                           [(0, 0), (w, 0), (w, h), (0, h)])
    cv2.imshow("perspect plate", plate)




    # contourList = opencvlib.GetOuterContours(mask, 100);
    # contourList = opencvlib.CleanContours(mask, contourList, 0.2, 0.98);
    # repairCont = opencvlib.GetRepairContour(contourList[0],4)
    # if(repairCont is not None):
    #     mask = opencvlib.GetContourRegionMask(mask,repairCont)
    # else:
    #     mask = opencvlib.GetContourRegionMask(mask, contourList)
    # cv2.imshow("clean small contour mask", mask)
    #
    #
    #
    #
    #
    # caliAngle = opencvlib.RotatedImgCalibrate(mask);
    # if(caliAngle!=0):
    #     print("rotation",caliAngle)
    #     rotatePlate = opencvlib.Rotate(plate,caliAngle)
    #     mask = opencvlib.Rotate(mask, caliAngle)
    #     maskand = cv2.bitwise_and(rotatePlate, rotatePlate, mask=mask)
    #     cv2.imshow("rotate Mask and", maskand)
    #     mask = opencvlib.GetOtusImage(maskand)[1]############
    #     cv2.imshow("wwwa", mask)################
    #     vertices = opencvlib.FindRotatedImgClimaxes(mask)
    #     topleft, topright, botright, botleft = vertices
    #     plate = opencvlib.PerspectiveTransform(rotatePlate,
    #                                            [topleft, topright, botright, botleft],
    #                                            [(0, 0), (w, 0), (w, h), (0, h)])
    # else:
    #     maskand = cv2.bitwise_and(plate, plate, mask=mask)
    #     cv2.imshow("rotate Mask and", maskand)
    #     mask = opencvlib.GetOtusImage(maskand)[1]##################
    #     cv2.imshow("wwwa", mask)################
    #     vertices = opencvlib.FindRotatedImgClimaxes(mask)
    #     topleft, topright, botright, botleft = vertices
    #     plate = opencvlib.PerspectiveTransform(plate,
    #                                            [topleft, topright, botright, botleft],
    #                                            [(0, 0), (w, 0), (w, h), (0, h)])
    # plate = opencvlib.Resize(plate,250,120)
    # cv2.imshow("perspect plate", plate)
    # mask = opencvlib.GetOtusImage(plate)[1]  ##################
    # cv2.imshow("perspect plate mask", mask)  ################


    # return plate;
""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # 过滤掉小块轮廓


    # rectContour = opencvlib.FindOptRectContour(contourList[0])
    # center, size, angle = cv2.minAreaRect(rectContour[0]);
    # print("angle",angle)
    # if(1<=angle<=88 or -88<=angle<=-1):
    #     # contourList = opencvlib.GetRepairContours(contourList, 0.02);
    #     contourList = rectContour
    #     # 修正矩形，说明本身板子有点歪，我们就取蓝色，再根据蓝色区域定4个顶点，做透视变换
    #     conMask = opencvlib.GetContourRegionMask(plate, contourList);
    #     cv2.imshow("conMask", conMask)
    #     maskand = cv2.bitwise_and(plate, plate, mask=conMask)
    #     ############################################
    #     cv2.imshow("mask and", maskand)
    #     ############################################
    #     hsvImg = cv2.cvtColor(maskand, cv2.COLOR_BGR2HSV);
    #     lower_blue = np.array([100, 44, 48]);
    #     upper_blue = np.array([133, 255, 255]);
    #     mask = cv2.inRange(hsvImg, lower_blue, upper_blue);
    #     res = cv2.bitwise_and(hsvImg, hsvImg, mask=mask);
    #     res = opencvlib.GetGrayImage(res)
    #     res = opencvlib.HistGlobalEqualize(res)
    #     res = opencvlib.GetOtusImage(res)[1]
    #     cv2.imshow("res", res)
    #
    #     vertices = opencvlib.FindRotatedImgClimaxes(res)
    #     topleft, topright, botright, botleft = vertices
    #     plate = opencvlib.PerspectiveTransform(plate,
    #                                            [topleft, topright, botright, botleft],
    #                                            [(0, 0), (w, 0), (w, h), (0, h)])
    #     cv2.imshow("repair", plate)
    # else:
    #     contourList = rectContour;
    #     # 没修正矩形的话，板子本来就很正，直接取矩形区域的四个点，进行透视变换
    #     conMask = opencvlib.GetContourRegionMask(plate, contourList);
    #     cv2.imshow("conMask", conMask)
    #     maskand = cv2.bitwise_and(plate, plate, mask=conMask)
    #     ############################################
    #     cv2.imshow("mask and", maskand)
    #     ############################################
    #     # hsvImg = cv2.cvtColor(maskand, cv2.COLOR_BGR2HSV);
    #     # lower_blue = np.array([100, 44, 48]);
    #     # upper_blue = np.array([133, 255, 255]);
    #     # mask = cv2.inRange(hsvImg, lower_blue, upper_blue);
    #     # res = cv2.bitwise_and(hsvImg, hsvImg, mask=mask);
    #     # res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR);
    #     # res = opencvlib.GetGrayImage(res)
    #     # res = opencvlib.HistGlobalEqualize(res)
    #     # res = opencvlib.GetOtusImage(res)[1]
    #     res = conMask
    #     cv2.imshow("res",res)
    #
    #     vertices = opencvlib.FindRotatedImgClimaxes(res)
    #     topleft, topright, botright, botleft = vertices
    #     # topleft = opencvlib.TopLeftCornerScan(res)
    #     # topright = opencvlib.TopRightCornerScan(res)
    #     # botleft = opencvlib.BotLeftCornerScan(res)
    #     # botright = opencvlib.BotRightCornerScan(res)
    #     plate = opencvlib.PerspectiveTransform(plate,
    #                                            [topleft, topright, botright, botleft],
    #                                            [(0, 0), (w, 0), (w, h), (0, h)])
    #     cv2.imshow("no repair", plate)




    # finalContour = None;
    # alal = 0;
    # for ep in np.arange(0.0, 0.9, 0.01):
    #     for c in contourList:
    #         repairContList = opencvlib.GetRepairContours(c, ep);
    #         conMask = opencvlib.GetContourRegionMask(plate, repairContList);
    #         x = np.where(conMask > 0)[::-1][0]
    #         y = np.where(conMask > 0)[::-1][1]
    #         roi_x1 = np.min(x)
    #         roi_x2 = np.max(x)
    #         roi_y1 = np.min(y)
    #         roi_y2 = np.max(y)
    #         potentialPlate = img[roi_y1:roi_y2, roi_x1:roi_x2];
    #         cv2.imshow("bbbbbb"+str(alal),potentialPlate);
    #         alal += 1
    #         if(IsPlate(potentialPlate)):
    #             finalContour = c;
    #             break;
    #     if(finalContour is not None):
    #         break;
    # print("final cpnt", finalContour)



    # 透视变换坐标匹配准备，从rectContourPoints里找到距离图像框四角最近的各 坐标对
    # rectContourPoints = opencvlib.FindOptRectContour(contourList[0]);
    # corners = rectContourPoints[0]
    # dist = [];
    # imgCorners = [(0, 0), (w, 0), (w, h), (0, h)];
    # for idx1 in range(len(imgCorners)):
    #     min = 1000000;
    #     x = 0;
    #     for idx2, _ in enumerate(corners):
    #         tmpDist = opencvlib.Distance(corners[idx2], imgCorners[idx1])
    #         if (min > tmpDist):
    #             min = tmpDist;
    #             x = idx2;
    #     dist.append([imgCorners[idx1], corners[x]]);
    #
    # # 进行透视变换
    # plate = opencvlib.PerspectiveTransform(plate,
    #                                        [dist[0][1], dist[1][1], dist[2][1], dist[3][1]],
    #                                        [dist[0][0], dist[1][0], dist[2][0], dist[3][0]])










print("************ start looking for plate region candidates *************")
interest_roi_list, roi_coord_list = findPlates(img);
if(interest_roi_list is not None):
    tempList = [];
    tempList_2 = [];
    for idx,eachROI in enumerate(interest_roi_list):
        rois, _ = findPlates(eachROI,
                         minAreaRatio2del=0.1,
                         maxAreaRatio2del=0.9,
                         resize_w=200,
                         resize_h=int(200 * plate_w_h_ratio),
                         veri_edges_min=35,
                         hori_edges_max=70);
        # cv2.imshow("aaaaaa",rois[0])
        if (len(rois) > 0 and IsPlate(rois[0])):
            tempList.append(rois[0]);
            tempList_2.append(roi_coord_list[idx]);
    interest_roi_list = tempList;
    roi_coord_list = tempList_2;
    del tempList;
    del tempList_2;
else:
    interest_roi_list = [];




if(len(interest_roi_list)>0):
    roi_x1 = roi_coord_list[0][0]
    roi_x2 = roi_coord_list[0][1]
    roi_y1 = roi_coord_list[0][2]
    roi_y2 = roi_coord_list[0][3]+5
    plateImg = img[roi_y1:roi_y2, roi_x1:roi_x2];
    plateImg = opencvlib.Resize(plateImg,300,150)


    plate = PerspectivePlate(plateImg)
    # cv2.imshow("Perspective plate", plateImg);
    # plate = PerspectivePlate(plate)
    # cv2.imshow("Perspective plate", plate);



else:
    plateImg = None;
    print("can not find plates");





print("DONE");




opencvlib.WaitEscToExit();
cv2.destroyAllWindows();


import cv2
import numpy as np
import opencvlib


def CleanMask(mask,minArea,maxArea):
    return opencvlib.GetContourRegionMask(mask,
                opencvlib.CleanContours(mask, opencvlib.GetOuterContours(mask, 100),
                minArea, maxArea));




def ExpandMask(mask):
    horiz_kernel = opencvlib.GetAreaKernel(cv2.MORPH_RECT, 1, 5);
    veri_kernel = opencvlib.GetAreaKernel(cv2.MORPH_RECT, 3, 1);
    h, w = mask.shape[0:2]
    blankImg = opencvlib.GetBlankImg(w, h)
    blankImg[:, 0:w // 2] = mask[:, 0:w // 2]
    left = cv2.dilate(blankImg, horiz_kernel, iterations=1, anchor=(4, 0));
    blankImg[:, :] = 0
    blankImg[:, w // 2:w] = mask[:, w // 2:w]
    right = cv2.dilate(blankImg, horiz_kernel, iterations=1, anchor=(0, 0));
    # blankImg[:, :] = 0
    # blankImg[0:h // 2, :] = mask[0:h // 2, :]
    # top = cv2.dilate(blankImg, veri_kernel, iterations=1, anchor=(0, 2));
    # blankImg[:, :] = 0
    # blankImg[h // 2:h, :] = mask[h // 2:h, :]
    # bottom = cv2.dilate(blankImg, veri_kernel, iterations=1, anchor=(0, 0));
    mask = cv2.bitwise_or(left, right);
    # mask = cv2.bitwise_or(mask, top);
    # mask = cv2.bitwise_or(mask, bottom);
    return mask











img = cv2.imread("../chepai/20.jpg")
img = opencvlib.Resize(img,500,450)
# img = opencvlib.HistGlobalEqualize(img)
cv2.imshow("img", img)

otus = opencvlib.GetOtusImage(img)[1]
cv2.imshow("otusimg", otus)

h,w = img.shape[0:2];
imgArea = h*w
minAreaThresh = int(imgArea*0.0005);
maxAreaThresh = int(imgArea*0.01);









# b,g,r = cv2.split(img)
# b = opencvlib.HistGlobalEqualize(b)
# g = opencvlib.HistGlobalEqualize(g)
# r = opencvlib.HistGlobalEqualize(r)
# # red MSER
# red_regions = opencvlib.MSER(r,minArea2del=minAreaThresh
#                          ,maxArea2del=maxAreaThresh
#                          ,minDiversity=0.5
#                          ,maxVariation=0.12);
# red_MSER = opencvlib.GetBlankImg(w,h);
# for reg in red_regions:
#     for point in reg:
#         row = point[1];
#         col = point[0];
#         if(red_MSER[row, col] == 0):
#             red_MSER[row, col] = 255;
# cv2.imshow("red_MSER", red_MSER);
#
#
#
# # green MSER
# green_regions = opencvlib.MSER(g,minArea2del=minAreaThresh
#                          ,maxArea2del=maxAreaThresh
#                          ,minDiversity=0.5
#                          ,maxVariation=0.12);
# green_MSER = opencvlib.GetBlankImg(w, h);
# for reg in green_regions:
#     for point in reg:
#         row = point[1];
#         col = point[0];
#         if (green_MSER[row, col] == 0):
#             green_MSER[row, col] = 255;
# cv2.imshow("green_MSER", green_MSER);
#
#
#
#
# # green MSER
# blue_regions = opencvlib.MSER(b,minArea2del=minAreaThresh
#                          ,maxArea2del=maxAreaThresh
#                          ,minDiversity=0.5
#                          ,maxVariation=0.12);
# blue_MSER = opencvlib.GetBlankImg(w, h);
# for reg in blue_regions:
#     for point in reg:
#         row = point[1];
#         col = point[0];
#         if (blue_MSER[row, col] == 0):
#             blue_MSER[row, col] = 255;
# cv2.imshow("blue_MSER", blue_MSER)
#
#
# mask = cv2.bitwise_and(red_MSER,green_MSER);
# mask = cv2.bitwise_and(mask,blue_MSER);
# cv2.imshow("and res",mask)
#
#
# contList = opencvlib.GetContours(mask,100)
# oriImg, contImg = opencvlib.DrawAllContoursOnImage(img,contList)
# cv2.imshow("contImg", contImg)




# gray MSER


gray = opencvlib.GetGrayImage(img);
gray = opencvlib.HistGlobalEqualize(gray)
gray_regions = opencvlib.MSER(gray,minArea2del=minAreaThresh
                         ,maxArea2del=maxAreaThresh
                         ,minDiversity=0.5
                         ,maxVariation=0.12);
gray_MSER = opencvlib.GetBlankImg(w,h);
for reg in gray_regions:
    for point in reg:
        row = point[1];
        col = point[0];
        if(gray_MSER[row, col] == 0):
            gray_MSER[row, col] = 255;
cv2.imshow("gray_MSER", gray_MSER);



maskand = cv2.bitwise_and(otus,gray_MSER)
cv2.imshow("maskand",maskand)

contList = opencvlib.GetOuterContours(maskand,100)
contList = opencvlib.CleanContoursByArea(contList,10,500*450)
contList = opencvlib.CleanContoursByEllipseAxisRatio(contList,1,6)

cleanMask = opencvlib.GetContourRegionMask(img,contList)
cv2.imshow("cleanMask", cleanMask)

for i,c in enumerate(contList):
    rectCont = opencvlib.FindOptRectContour(c);
    opencvlib.DrawAllContoursOnImage(img,rectCont,drawPointSize=2);
cv2.imshow("contImg", img)












# rectList = []
# sameRectConts = []
# for i,c in enumerate(contList):
    # mask = opencvlib.GetContourRegionMask(img, c)
    # cv2.imshow("cont"+str(i),mask)
    # rectCont = opencvlib.FindOptRectContour(c);
#
#     if(np.all(rectCont) not in rectList):
#         rectList.append(rectCont)
#         index = rectList.index(rectCont);
#         sameRectConts.append([c])
#     else:
#         index = rectList.index(rectCont);
#         sameRectConts[index].append(c)
#
#
# for contList in sameRectConts:
#     mask = opencvlib.GetContourRegionMask(img,c)
#     cv2.imshow("cont"+str(i),mask)














# kernel_33 = opencvlib.GetAreaKernel(cv2.MORPH_RECT,3,3);
# horiz_kernel = opencvlib.GetAreaKernel(cv2.MORPH_RECT, 1, 3);
# vert_kernel = opencvlib.GetAreaKernel(cv2.MORPH_RECT, 3, 1);
#
# vert_erode = cv2.erode(mask, vert_kernel, iterations=1);
# cv2.imshow("vert_erode", vert_erode);




# sobelx_roi = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1);
# sobelx_roi = np.uint8(np.absolute(sobelx_roi))
# sobelx_roi = opencvlib.GetOtusImage(sobelx_roi)[1]
# cv2.imshow("sobelx_roi", sobelx_roi)
#
#
#
# # mask_and = cv2.bitwise_and(img,img,mask=vert_erode)
# # mask_and = cv2.bitwise_and(mask_and,mask_and,mask=sobelx_roi)
# mask_and = cv2.bitwise_and(vert_erode,sobelx_roi);
# # mask_and = CleanMask(mask_and,0.0001,1)
# cv2.imshow("mask_and", mask_and)
#
# maskQ = mask_and;
# while(True):
#     maskQ = ExpandMask(maskQ);
#     maskQ = cv2.erode(maskQ, vert_kernel, iterations=1);
#     maskQ = cv2.dilate(maskQ, vert_kernel, iterations=1);
#     # maskQ = cv2.erode(maskQ, horiz_kernel, iterations=1);
#     # maskQ = cv2.dilate(maskQ, horiz_kernel, iterations=1);
#
#
# # maskQ = ExpandMask(mask_and);
# cv2.imshow("maskQ", maskQ)
#
# contList = opencvlib.GetOuterContours(maskQ,100);
# for i,c in enumerate(contList):
#     cont = opencvlib.FindOptRectContour(c);
#     oriImg, contImg = opencvlib.DrawAllContoursOnImage(img,cont)
# cv2.imshow("contImg", contImg)


# medianBlur = cv2.medianBlur(mask_and,3);
# cv2.imshow("medianBlur", medianBlur)



# blur = cv2.GaussianBlur(mask_and,(3,3),0);
# diff = mask_and - blur
# cv2.imshow("diff", diff)



# vert_erode = cv2.erode(mask_and,vert_kernel,iterations=1)
# cv2.imshow("vert_erode", vert_erode)


# hori_erode = cv2.erode(vert_erode,horiz_kernel,iterations=1)
# cv2.imshow("erode", hori_erode)





opencvlib.WaitEscToExit()
cv2.destroyAllWindows()
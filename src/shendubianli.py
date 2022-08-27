####计算最大连通域面积、连通域面积列表、画出连通域轮廓、
###删除小面积连通域

import numpy as np
from PIL import Image
from skimage import data,filters,segmentation,measure,morphology
import os
import cv2
import matplotlib.pyplot as plt


threshold = 500
###########找到最大连通域面积###################
def get_max(contours):
    sum = list()
    SUM1 = 1
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])#计算轮廓面积，但是可能和像素点区域不太一样
        # area = cv2.countNonZero(img3)  # 输入图像是单通道 计算像素点个数
        sum.append(area)
        sum.sort() #列表值从小到大排序
        SUM1 = sum[-1] #sum1总是目标面积最大值
    return SUM1

###########计算连通域面积###################
def get_yu(contours):
    sum = list()

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])#计算轮廓面积，但是可能和像素点区域不太一样 收藏
        # area = cv2.countNonZero(contours[i])  # 输入图像是单通道 计算像素点个数
        sum.append(area)
        sum.sort() #列表值从小到大排序
    return sum
'''
#  遍历文件里的所有
img_dir = 'E:\\aaaaaaaaaaaa\\try\\1\\'
for filename in os.listdir(img_dir):
    img_name = img_dir + filename
'''


# 下面三句效果等同于    img = Image.open(img_name)    thresh = np.array(img)
    # img = cv2.imread(img_name)
    # img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(img, 0, 1,cv2.THRESH_BINARY)  # 灰度图二值


#img_name
img = cv2.imread('output/sample1_erzhi.png', -1)
#img = Image.open(img_name)
thresh = np.array(img)

######cv2.findContours. opencv3版本会返回3个值，opencv2和4只返回后两个值
img2, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#img3是返回的二值图像，contours返回的是轮廓像素点列表，一张图像有几个目标区域就有几个列表值

for i in range(len(contours)):
     area = cv2.contourArea(contours[i])#计算轮廓面积，但是可能和像素点区域不太一样 收藏
     ###判断轮廓面积是否小于阈值，小于阈值就删除连通域
     if area < threshold:
        cv2.drawContours(img2, [contours[i]], 0, 0, -1)

cv2.imwrite('output/zuidayu.png', img2)

# ############白线画轮廓
#     # img2 = cv2.imread('E:\\aaaaaaaaaaaa\\try\\1\\label1.png')
#     # img3 = cv2.drawContours(img2, contours, -1,(255,255,255),1)
#     # cv2.imwrite('E:\\aaaaaaaaaaaa\\try\\1\\label11.png',img3)

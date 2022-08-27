import glob
from PIL import Image
import numpy as np
import cv2
import os

#-------------------将图片转为rgb格式-------------
img_path ='output/sample1.png'

for jpg_path in glob.glob(img_path):
    img_name = os.path.basename(jpg_path)
    img = cv2.imread(jpg_path,0)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.imwrite('output/sample1_rgb.png', img_rgb)

# ------------------抠图--------------------------
pth = "output/sample1_rgb.png"
img = Image.open(pth)
img1 = cv2.imread(pth)
mask = cv2.imread('output/zuidayu.png', 0)  # 读取灰度图像
height, width, channel = img1.shape
b, g, r = cv2.split(img1)

# -----------------获取透明前景图像-----------
dstt = np.zeros((4, height, width), dtype=img1.dtype)
dstt[0][0:height, 0:width] = b
dstt[1][0:height, 0:width] = g
dstt[2][0:height, 0:width] = r
dstt[3][0:height, 0:width] = mask
# cv2.imwrite("output/" + imgname + "_mat1.png", cv2.merge(dstt))

# -----------------加入背景图像-----------
bg = np.zeros((3, height, width), dtype=img1.dtype)  # 生成背景图像
bg[0][0:height, 0:width] = 0
bg[1][0:height, 0:width] = 0
bg[2][0:height, 0:width] = 0
# 背景图像采用白色

dstt = np.zeros((3, height, width), dtype=img1.dtype)

for i in range(3):
    dstt[i][:, :] = bg[i][:, :] * (255.0 - mask) / 255
    dstt[i][:, :] += np.array(img1[:, :, i] * (mask / 255), dtype=np.uint8)
cv2.imwrite("output/sample1_mat1.png", cv2.merge(dstt))


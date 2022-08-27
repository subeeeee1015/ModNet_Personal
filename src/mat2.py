from PIL import Image
import cv2
import numpy as np


#-------------------将图片转为灰度格式-------------
path_mat2 = 'output/sample1_mat1.png'
img = cv2.imread(path_mat2)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('output/sample1_mat2.png', gray)

# ------------------二次抠图--------------------------
pth = "example/sample1.jpg"
img = Image.open(pth)
img1 = cv2.imread(pth)
mask = cv2.imread('output/sample1_mat2.png', 0)  # 读取灰度图像
height, width, channel = img1.shape
b, g, r = cv2.split(img1)

# -----------------获取透明前景图像-----------
dstt = np.zeros((4, height, width), dtype=img1.dtype)
dstt[0][0:height, 0:width] = b
dstt[1][0:height, 0:width] = g
dstt[2][0:height, 0:width] = r
dstt[3][0:height, 0:width] = mask
cv2.imwrite("output/sample1_mat3.png", cv2.merge(dstt))

# -----------------加入背景图像-----------
bg = np.zeros((3, height, width), dtype=img1.dtype)  # 生成背景图像
bg[0][0:height, 0:width] = 100
bg[1][0:height, 0:width] = 245
bg[2][0:height, 0:width] = 98
# 背景图像采用bai色

dstt = np.zeros((3, height, width), dtype=img1.dtype)

for i in range(3):
    dstt[i][:, :] = bg[i][:, :] * (255.0 - mask) / 255
    dstt[i][:, :] += np.array(img1[:, :, i] * (mask / 255), dtype=np.uint8)
cv2.imwrite("output/sample1_mat4.png", cv2.merge(dstt))


import cv2
import numpy as np
from PIL import Image
import os

pth = "output/sample1_zuida.png"
img = Image.open(pth)
img1 = cv2.imread(pth)
imgname_1 = os.path.basename(pth)
imgname_2 = os.path.splitext(imgname_1)
imgname = imgname_2[0]
# img1 = cv2.imread("output/sample1.png")
kernel = np.ones(shape=[3, 3], dtype=np.uint8)
img2 = cv2.dilate(img1, kernel, iterations=6)
cv2.imwrite("output/"+imgname+"_dilate.png", img2)

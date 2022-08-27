import cv2
import os
import numpy as np
from pathlib import Path
import time
from PIL import Image
import numpy as np



img_open = Image.open('output/sample1_open.png')
img_open = np.array(img_open)
#标注图
img_alpha = Image.open('output/sample1.png')
img_alpha = np.array(img_alpha)

intersection = np.sum(np.logical_and(img_open, img_alpha))
union = np.sum(np.logical_or(img_open, img_alpha))

# Intersection over Union
iou_img = intersection / union
print(iou_img)

if iou_img < 0.95:
    for x in range(20):
        kernel = np.ones(shape=[3, 3], dtype=np.uint8)
        img_dilate = cv2.dilate(img_open, kernel, iterations=x)
        img_open = img_dilate
        intersection = np.sum(np.logical_and(img_open, img_alpha))
        union = np.sum(np.logical_or(img_open, img_alpha))
        iou_img = intersection / union
        if iou_img > 0.95:
            print(x)
            cv2.imwrite("output/" + imgname + "_dilate.png", img_dilate)
            break
print(iou_img)

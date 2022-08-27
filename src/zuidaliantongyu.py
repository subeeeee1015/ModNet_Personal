import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# read the image
img_max = cv2.imread("output/sample1_erzhi.png")
gray_max = cv2.cvtColor(img_max, cv2.COLOR_BGR2GRAY)

# take the  threshold and morphology thransform
ret, binary = cv2.threshold(gray_max, 129, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
bin_clo = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

# obtain th label of the connection areas
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)

"""
#查看各个返回值
print('num_labels = ',num_labels)
print('stats = ',stats)
print('centroids = ',centroids)
print('labels = ',labels)
"""

label_area = stats[:, -1]
max_index = np.argmax(label_area)

# label the backgroud and foreground
height1 = labels.shape[0]
width1 = labels.shape[1]
for row in range(height1):
    for col in range(width1):
        if labels[row, col] == max_index:
            gray_max[row, col] = 255
        else:
            gray_max[row, col] = 0
        # if stats[labels[row,col],4] < 100:
        # gray[row,col] = 255

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
conne = cv2.morphologyEx(gray_max, cv2.MORPH_OPEN, kernel, iterations=2)
# cv2.namedWindow('results', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('results', conne)
# cv2.waitKey()

# image_gray = cv2.cvtColor(conne, cv2.COLOR_BGR2GRAY)
cv2.imwrite('output/sample1_zuida.png', conne)



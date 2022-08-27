import cv2

img_result = cv2.imread('output/sample3.png',-1)
cv2.namedWindow('results', cv2.WINDOW_KEEPRATIO)
cv2.imshow('results', img_result)
cv2.waitKey()
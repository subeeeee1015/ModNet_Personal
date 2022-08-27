from PIL import Image
import cv2

#等比例缩放

def threshold_By_OTSU(img_alpha):
    gray = cv2.cvtColor(img_alpha, cv2.COLOR_BGR2GRAY)   ##要二值化图像，必须先将图像转为灰度图
    ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY) ##第2空的数字是阈值
    return binary
if __name__ == '__main__':
    img = cv2.imread('output/sample1.png')
    img = threshold_By_OTSU(img)
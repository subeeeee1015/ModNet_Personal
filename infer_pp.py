from src.models.modnet import MODNet
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import os

def predit_matte(modnet: MODNet, im: Image):
    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # define hyper-parameters
    ref_size = 512

    modnet.eval()

    # unify image channels to 3
    im = np.asarray(im)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    im = Image.fromarray(im)
    # convert image to PyTorch tensor
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    return matte

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

def get_yu(contours):
    sum = list()

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])#计算轮廓面积，但是可能和像素点区域不太一样 收藏
        # area = cv2.countNonZero(contours[i])  # 输入图像是单通道 计算像素点个数
        sum.append(area)
        sum.sort() #列表值从小到大排序
    return sum

def threshold_pp(img_alpha):
    gray = cv2.cvtColor(img_alpha, cv2.COLOR_BGR2GRAY)   ##要二值化图像，必须先将图像转为灰度图
    ret, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY) ##第2空的数字是阈值
    return binary

def infer_pp(pth):
    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    ckp_pth = 'pretrained/modnet_photographic_portrait_matting.ckpt'
    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(ckp_pth)
    else:
        weights = torch.load(ckp_pth, map_location=torch.device('gpu'))
    modnet.load_state_dict(weights)

    img = Image.open(pth)


    matte = predit_matte(modnet, img)
    prd_img = Image.fromarray(((matte * 255).astype('uint8')), mode='L')

    img1 = cv2.imread(pth)
    imgname_1 = os.path.basename(pth)
    imgname_2 = os.path.splitext(imgname_1)
    imgname = imgname_2[0]
    prd_img.save('output/'+imgname+'.png')
    mask = cv2.imread("output/" + imgname+'.png', 0)  # 读取灰度图像
    height, width, channel = img1.shape
    b, g, r = cv2.split(img1)

# -----------------获取透明的前景图像-----------
    dstt = np.zeros((4, height, width), dtype=img1.dtype)

    dstt[0][0:height, 0:width] = b
    dstt[1][0:height, 0:width] = g
    dstt[2][0:height, 0:width] = r
    dstt[3][0:height, 0:width] = mask
    img_fore = cv2.merge(dstt)
    cv2.imwrite("output/"+imgname+"_fore.png", img_fore)

    # 二值化图片
    img_erzhi = cv2.imread('output/'+ imgname +'.png')
    img_erzhi = threshold_pp(img_erzhi)


# ---------------------找最大区域------------------------
    thresh = np.array(img_erzhi)
######cv2.findContours. opencv3版本会返回3个值，opencv2和4只返回后两个值
    img3, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#img3是返回的二值图像，contours返回的是轮廓像素点列表，一张图像有几个目标区域就有几个列表值
    num_array = len(contours)
    if num_array == 0:
        print('找不到目标,跳过当前帧')
    else:
        thr_array = [0 for i in range(num_array)]
        for i in range(num_array):
            thr_array[i] = cv2.contourArea(contours[i])
            # 计算轮廓面积，但是可能和像素点区域不太一样
        threshold = thr_array[0]
        for i in range(num_array):
            if thr_array[i] > threshold:
                threshold = thr_array[i]
        for i in range(num_array):
            if thr_array[i] < threshold:
                img3 = cv2.drawContours(img3, [contours[i]], 0, 0, -1)

    # -----------判断轮廓面积是否小于阈值，将最大域保存到threshold然后进行比较，比thr小的就删除----------------
#-------------------将图片转为rgb格式-------------
    img4 = cv2.imread('output/'+imgname+'.png', 0)
    img_rgb = cv2.cvtColor(img4,cv2.COLOR_GRAY2BGR)

# ------------------抠图--------------------------

# img = Image.open(pth)
    img5 = img_rgb
    mask = img3  # 读取灰度图像
    height, width, channel = img5.shape
    b, g, r = cv2.split(img5)

# -----------------加入背景图像-----------
    dstt = np.zeros((4, height, width), dtype=img5.dtype)
    dstt[0][0:height, 0:width] = b
    dstt[1][0:height, 0:width] = g
    dstt[2][0:height, 0:width] = r
    dstt[3][0:height, 0:width] = mask

    bg = np.zeros((3, height, width), dtype=img5.dtype)  # 生成背景图像
    bg[0][0:height, 0:width] = 0
    bg[1][0:height, 0:width] = 0
    bg[2][0:height, 0:width] = 0
# 背景图像采用白色

    dstt = np.zeros((3, height, width), dtype=img5.dtype)

    for i in range(3):
        dstt[i][:, :] = bg[i][:, :] * (255.0 - mask) / 255
        dstt[i][:, :] += np.array(img5[:, :, i] * (mask / 255), dtype=np.uint8)

#----因为抠完图加入背景后，图片格式为rgb三通道图，需要转换成单通道的---
#----------------------将图片转为灰度格式----------------------

    img5 = cv2.merge(dstt)
    gray = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output/'+imgname+'_alpha.png', gray)

# -------------------------二次抠图--------------------------
    pth_result = 'example/'+ imgname_1
    # img6 = Image.open(pth_result)
    img6 = cv2.imread(pth_result)
    mask_result = gray
    height, width, channel = img6.shape
    b, g, r = cv2.split(img6)

# -----------------获取透明前景图像-----------
    dstt = np.zeros((4, height, width), dtype=img6.dtype)
    dstt[0][0:height, 0:width] = b
    dstt[1][0:height, 0:width] = g
    dstt[2][0:height, 0:width] = r
    dstt[3][0:height, 0:width] = mask_result
    cv2.imwrite('output/'+imgname+'_result.png', cv2.merge(dstt))
# ----------------添加纯色背景------------------
    bg = np.zeros((3, height, width), dtype=img6.dtype)  # 生成背景图像
    bg[0][0:height, 0:width] = 255
    bg[1][0:height, 0:width] = 255
    bg[2][0:height, 0:width] = 255
    # 背景图像采用规定色

    dstt = np.zeros((3, height, width), dtype=img6.dtype)

    for i in range(3):
        dstt[i][:, :] = bg[i][:, :] * (255.0 - mask_result) / 255
        dstt[i][:, :] += np.array(img6[:, :, i] * (mask_result / 255), dtype=np.uint8)
    cv2.imwrite("output/" + imgname + "_result1.png", cv2.merge(dstt))

    img_result1 = cv2.imread('output/'+imgname+'_result1.png')
    cv2.namedWindow('results', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('results', img_result1)
    cv2.waitKey()
    return cv2.merge(dstt)

if __name__ == '__main__':
    pth = 'example/sample2.jpg'
    infer_pp(pth)




from models.modnet import MODNet
from PIL import Image
import numpy as np
from torchvision import transforms
from skimage.measure import label
import torch
import torch.nn.functional as F
import torch.nn as nn
import glob
import cv2
import os
from matplotlib import pyplot as plt

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


if __name__ == '__main__':
    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    ckp_pth = './pretrained/modnet_photographic_portrait_matting.ckpt'
    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(ckp_pth)
    else:
        weights = torch.load(ckp_pth, map_location=torch.device('gpu'))
    modnet.load_state_dict(weights)

    pth = 'example/sample6.jpg'

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

# -----------------1.获取透明的前景图像-----------
    dstt = np.zeros((4, height, width), dtype=img1.dtype)

    dstt[0][0:height, 0:width] = b
    dstt[1][0:height, 0:width] = g
    dstt[2][0:height, 0:width] = r
    dstt[3][0:height, 0:width] = mask
    toumingqianjing = cv2.merge(dstt)
    cv2.imwrite("output/"+imgname+"_fore.png", toumingqianjing)

# --------------3.计算阈值-----------------------
    image = cv2.imread("output/" + imgname + ".png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
#方法选择为THRESH_OTSU
    # print(ret1)

# -------------4.计算最大连通域-------------------
# 读取图片
    img_max = image
    gray_max = cv2.cvtColor(img_max, cv2.COLOR_BGR2GRAY)

# 决定阈值and形态
    ret, binary = cv2.threshold(gray_max, ret1, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_clo = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# 获取连接区域的标签
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)
    label_area = stats[:, -1]
    max_index = np.argmax(label_area)

# 标记背景和前景
    height1 = labels.shape[0]
    width1 = labels.shape[1]
    for row in range(height1):
        for col in range(width1):
            if labels[row, col] == max_index:
                gray_max[row, col] = 255
            else:
                gray_max[row, col] = 0

# cv2.imwrite("output/"+imgname+"_max.png", gray_max)

# 形态学变换开运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    conne = cv2.morphologyEx(gray_max, cv2.MORPH_OPEN, kernel, iterations=3)
# cv2.imwrite("output/"+imgname+"_open.png", conne)

# ---------------计算损失------------------
# 读取膨胀图
    img_open = conne
    img_open = np.array(img_open)
# 标注遮罩图
    img_alpha = Image.open('output/' + imgname + '.png')
    img_alpha = np.array(img_alpha)

    intersection = np.sum(np.logical_and(img_open, img_alpha))
    union = np.sum(np.logical_or(img_open, img_alpha))

# Intersection over Union
    iou_img = intersection / union
    print(iou_img)

# ----------损失低于0.95则进行膨胀-------


    for x in range(10):
    # --------膨胀----------------
        img_open1 = img_open
        kernel = np.ones(shape=[3, 3], dtype=np.uint8)
        img_dilate = cv2.dilate(img_open1, kernel, iterations=x)
        img_open1 = img_dilate
        intersection = np.sum(np.logical_and(img_open1, img_alpha))
        union = np.sum(np.logical_or(img_open1, img_alpha))
        iou_img1 = intersection / union
        if iou_img1 > iou_img:
            iou_img = iou_img1
            x_best = x
            img_open2 = img_open1
        else:
            continue
    img_dilate = cv2.dilate(img_open2, kernel, iterations=x_best)
    cv2.imwrite("output/" + imgname + "_dilate.png", img_dilate)
    print("iou：", iou_img)
    print("x:",x_best)


# ------------6.对alpha图进行抠图---------------
# 先把alpha图转化成rgb图像
    imgmat_path ='output/' + imgname + '.png'
    for jpgmat_path in glob.glob(imgmat_path):
        imgmat = cv2.imread(jpgmat_path, 0)
        img_rgb = cv2.cvtColor(imgmat, cv2.COLOR_GRAY2BGR)

# 抠图
    mask = img_dilate  # 读取灰度图像
    # mask = cv2.imread('output/' + imgname + '_dilate.png', 0)  # 读取灰度图像
    height, width, channel = img_rgb.shape
    b, g, r = cv2.split(img_rgb)

    # 生成背景图像
    bg = np.zeros((3, height, width), dtype=img_rgb.dtype)
    bg[0][0:height, 0:width] = 0
    bg[1][0:height, 0:width] = 0
    bg[2][0:height, 0:width] = 0
    # 背景图像采用黑色为了后期转成灰度图可以一致

    dstt = np.zeros((3, height, width), dtype=img_rgb.dtype)

    for i in range(3):
        dstt[i][:, :] = bg[i][:, :] * (255.0 - mask) / 255
        dstt[i][:, :] += np.array(img_rgb[:, :, i] * (mask / 255), dtype=np.uint8)
    img_mat1 = cv2.merge(dstt)

    #-------------------将图片转为灰度格式-------------

    gray_mat1 = cv2.cvtColor(img_mat1, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output/' + imgname + '_mat.png', gray_mat1)

    # ------------------二次抠图--------------------------
    pth = 'example/' + imgname + '.jpg'
    img = Image.open(pth)
    img1 = cv2.imread(pth)
    mask = gray_mat1   #   读取灰度图
    #mask = cv2.imread('output/sample1_mat.png', 0)  # 读取灰度图像
    height, width, channel = img1.shape
    b, g, r = cv2.split(img1)

    # -----------------重新获取透明前景图像-----------
    dstt = np.zeros((4, height, width), dtype=img1.dtype)
    dstt[0][0:height, 0:width] = b
    dstt[1][0:height, 0:width] = g
    dstt[2][0:height, 0:width] = r
    dstt[3][0:height, 0:width] = mask
    cv2.imwrite("output/" + imgname + "_pp.png", cv2.merge(dstt))

    '''
    # -----------------加入背景图像-----------
    bg = np.zeros((3, height, width), dtype=img1.dtype)  # 生成背景图像
    bg[0][0:height, 0:width] = 100
    bg[1][0:height, 0:width] = 245
    bg[2][0:height, 0:width] = 98
    # 背景图像采用规定色
    
    dstt = np.zeros((3, height, width), dtype=img1.dtype)
    
    for i in range(3):
        dstt[i][:, :] = bg[i][:, :] * (255.0 - mask) / 255
        dstt[i][:, :] += np.array(img1[:, :, i] * (mask / 255), dtype=np.uint8)
    cv2.imwrite("output/" + imgname + "_result1.png", cv2.merge(dstt))
    
    '''
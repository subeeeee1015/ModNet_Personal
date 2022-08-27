from models.modnet import MODNet
from PIL import Image
import numpy as np
from torchvision import transforms
from skimage.measure import label
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


if __name__ == '__main__':
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

    pth = 'example/sample7.png'

    img = Image.open(pth)

    matte = predit_matte(modnet, img)
    prd_img = Image.fromarray(((matte * 255).astype('uint8')), mode='L')

    #prd_img.save('output/test_predic.jpg')


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
    cv2.imwrite("output/"+imgname+"_fore.png", cv2.merge(dstt))


# -----------------2.与新背景图像合成-----------
    bg = np.zeros((3, height, width), dtype=img1.dtype)  # 生成背景图像
    bg[0][0:height, 0:width] = 255
    bg[1][0:height, 0:width] = 255
    bg[2][0:height, 0:width] = 255
# 背景图像采用bai色

    dstt = np.zeros((3, height, width), dtype=img1.dtype)

    for i in range(3):
        dstt[i][:, :] = bg[i][:, :] * (255.0 - mask) / 255
        dstt[i][:, :] += np.array(img1[:, :, i] * (mask / 255), dtype=np.uint8)
    cv2.imwrite("output/"+imgname+"_merge.png", cv2.merge(dstt))



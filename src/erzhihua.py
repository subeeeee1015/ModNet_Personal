# 图片二值化
from PIL import Image

img = Image.open('output/sample1.png')

# 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
Img = img.convert('L')
#Img.save("pictures/学习资料11.jpg")

# 自定义灰度界限，大于这个值为黑色，小于这个值为白色
threshold = 10

table = []
for i in range(256):
    if i < threshold:
        table.append(0)
    else:
        table.append(1)

# 图片二值化 使用table来设置二值化的规则
photo = Img.point(table, '1')
photo.save("output/sample11.png")
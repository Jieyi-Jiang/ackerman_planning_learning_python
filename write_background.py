import cv2
import numpy as np

# 设置图片的宽度、高度和通道数（3表示彩色图像，1表示灰度图像）
width = 100
height = 100
channels = 3

# 创建一个纯白图片
# np.ones创建一个全1数组，然后乘以255得到纯白像素值
# 数据类型为uint8，表示像素值范围为0-255
white_image = np.ones((height, width, channels), dtype=np.uint8) * 255

# 保存图片
cv2.imwrite('white_image.jpg', white_image)

# 显示图片
cv2.imshow('White Image', white_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
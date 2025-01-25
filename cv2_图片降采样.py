import cv2
import numpy as np
# 读取图片
img = cv2.imread('./labyrinth_3.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = img[:,:,0] + img[:,:,1] +img[:,:,2]

# 定义新的尺寸
new_width = img.shape[1] // 5  # 宽度
new_height = img.shape[0] // 5  # 高度

img_gray[:, 0:5] = 0
img_gray[:, -5:] = 0
img_gray[0:5, :] = 0
img_gray[-5:, :] = 0

for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):
        threshold = 100
        if img_gray[i, j] > threshold:
            # negative_image[i, j] = (41, 40, 41)
            img_gray[i, j] = 255
        elif img_gray[i, j] < threshold:
            img_gray[i, j] = 0
# kernel = np.ones((41, 41), np.uint8)
# dilated_img = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel, iterations=10)

cv2.imshow('原图', img_gray)
# 使用cv2.resize进行降采样
resized_img = cv2.resize(img_gray, (new_width, new_height))

kernel = np.ones((3, 3), np.uint8)
# dilated_img = cv2.morphologyEx(resized_img, cv2.MORPH_CLOSE, kernel, iterations=5)
# resized_img = cv2.erode(resized_img, kernel, iterations=1)
for i in range(resized_img.shape[0]):
    for j in range(resized_img.shape[1]):
        threshold = 200
        if resized_img[i, j] > threshold:
            # negative_image[i, j] = (41, 40, 41)
            resized_img[i, j] = 255
        elif resized_img[i, j] < threshold:
            resized_img[i, j] = 0

# 保存降采样后的图片
cv2.imshow('降采样', resized_img)
cv2.imwrite('labyrinth_3_small.jpg', resized_img)
cv2.waitKey(0)

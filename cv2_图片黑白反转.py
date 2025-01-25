import cv2

# 读取图片
image = cv2.imread('labyrinth_2.png')
negative_image = image.copy()
# 检查图片是否成功读取
if image is None:
    print("Error: Could not open or find the image.")
else:
    # 计算负片效果

    # image = 255 - image
    for i in range(negative_image.shape[0]):
        for j in range(negative_image.shape[1]):
            threshold = 100
            if negative_image[i, j][0] > threshold:
                # negative_image[i, j] = (41, 40, 41)
                negative_image[i, j] = 0
            elif negative_image[i, j][0] < threshold:
                negative_image[i, j] = 255

    # 显示原始图片和负片效果图片
    # cv2.imshow('Original Image', image)
    cv2.imshow('Negative Image', negative_image)
    cv2.imwrite('img_1_reverse.png', negative_image)

    # 等待按键，然后关闭所有窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

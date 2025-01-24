import cv2

img1 = cv2.imread('./map_1.jpg')

img2 = cv2.GaussianBlur(img1, (21, 21), 0)
cv2.imshow('GaussianBlur', img2)
cv2.imwrite('map_2.jpg', img2)
cv2.waitKey(0)

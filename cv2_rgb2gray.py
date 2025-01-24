import cv2

img = cv2.imread('./map_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./map_1_gray.jpg', gray)

print(gray.shape)
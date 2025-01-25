import cv2
import numpy as np


img_path = './map3.jpg'
img_input = cv2.imread(img_path)
kernel = np.ones((2,2), dtype=np.uint8)
img_output = cv2.erode(img_input, kernel, iterations=1)

cv2.imshow('input', img_input)
cv2.imshow('output', img_output)
cv2.waitKey(0)
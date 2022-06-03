# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('candy.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred_Image = cv2.GaussianBlur(gray, (11, 11), 0)
cannyEdgeDetector = cv2.Canny(blurred_Image, 30, 150, 3)
dilation = cv2.dilate(cannyEdgeDetector, (1, 1), iterations=0)

(cnt, hierarchy) = cv2.findContours(
    dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgbColors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgbColors, cnt, -1, (0, 255, 0), 2)
cv2.imshow("Dilated", dilation)

print("coins in the image : ", len(cnt))
cv2.waitKey(0)
import cv2
import numpy as np

img = cv2.imread('a_Kaznu.jpg', 0)
kernel = np.ones((3, 3), np.uint8)

#original image
cv2.namedWindow("OriginalImage", cv2.WINDOW_NORMAL)
cv2.imshow('OriginalImage', img)
cv2.waitKey()

#threshold
th, img = cv2.threshold(img, 120,255, cv2.THRESH_BINARY)
cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
cv2.imshow('Threshold', img)
cv2.waitKey()

#erosion
erosion = cv2.erode(img, kernel, iterations=1)
cv2.namedWindow("Erosion", cv2.WINDOW_NORMAL)
cv2.imshow('Erosion', erosion)
cv2.waitKey()

#dilation
dilation = cv2.dilate(img, kernel, iterations=1)
cv2.namedWindow("Dilation", cv2.WINDOW_NORMAL)
cv2.imshow('Dilation', dilation)
cv2.waitKey()

#Opening is used for removing internal noise of the obtained image.
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.namedWindow("Opening", cv2.WINDOW_NORMAL)
cv2.imshow('Opening', opening)
cv2.waitKey()

#Closing is used for smoothening of contour and fusing of narrow breaks.
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.namedWindow("Closing", cv2.WINDOW_NORMAL)
cv2.imshow('Closing', closing)
cv2.waitKey()

#gradient (difference between dilation and erosion of an image.)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.namedWindow("Gradient", cv2.WINDOW_NORMAL)
cv2.imshow('Gradient', gradient)
cv2.waitKey()

#Top Hat. It is the difference between input image and Opening of the image
top_hat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.namedWindow("TopHat", cv2.WINDOW_NORMAL)
cv2.imshow('TopHat', top_hat)
cv2.waitKey()

#Black Hat. It is the difference between the closing of the input image and input image.
black_hat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.namedWindow("BlackHat", cv2.WINDOW_NORMAL)
cv2.imshow('BlackHat', black_hat)
cv2.waitKey()

cv2.destroyAllWindows()

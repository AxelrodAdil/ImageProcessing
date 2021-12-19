import cv2
import numpy as np

#read image
img = cv2.imread('sm_snes.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

#resize the image to demonstrate the functionality
scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
desiredResizedDimension = (width, height)

img = cv2.resize(img, desiredResizedDimension)

# run a sobel technique
sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow('Original', img)

cv2.namedWindow("Sobel Horizontal Filter", cv2.WINDOW_NORMAL)
cv2.imshow('Sobel Horizontal Filter', sobel_horizontal)

cv2.namedWindow("Sobel Vertical Filter", cv2.WINDOW_NORMAL)
cv2.imshow('Sobel Vertical Filter', sobel_vertical)

cv2.waitKey(0)

#now run a laplacian filter
laplacian_edges = cv2.Laplacian(img, cv2.CV_64F,ksize=5)

cv2.namedWindow("Laplacian Filter", cv2.WINDOW_NORMAL)
cv2.imshow('Laplacian Filter', laplacian_edges)

cv2.waitKey(0)

#now run a canny technique
canny_edges = cv2.Canny(img, 10, 10, edges=None, apertureSize=5)
cv2.namedWindow("Canny Edges", cv2.WINDOW_NORMAL)
cv2.imshow('Canny Edges', canny_edges)

cv2.waitKey()

cv2.destroyAllWindows()
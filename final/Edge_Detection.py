import cv2

#read image
img = cv2.imread('a_Kaznu.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

#percent of original size
scale_percent = 60
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
desiredResizedDimension = (width, height)
img = cv2.resize(img, desiredResizedDimension)

#sobel technique
#https://en.wikipedia.org/wiki/Sobel_operator
sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow('Original', img)
cv2.waitKey()

cv2.namedWindow("Sobel Horizontal Filter", cv2.WINDOW_NORMAL)
cv2.imshow('Sobel Horizontal Filter', sobel_horizontal)
cv2.waitKey()

cv2.namedWindow("Sobel Vertical Filter", cv2.WINDOW_NORMAL)
cv2.imshow('Sobel Vertical Filter', sobel_vertical)
cv2.waitKey()

#laplacian filter
laplacian_edges = cv2.Laplacian(img, cv2.CV_64F, ksize=5)

cv2.namedWindow("Laplacian Filter", cv2.WINDOW_NORMAL)
cv2.imshow('Laplacian Filter', laplacian_edges)
cv2.waitKey()

#canny technique
canny_edges = cv2.Canny(img, 10, 10, edges=None, apertureSize=5)
cv2.namedWindow("Canny Edges", cv2.WINDOW_NORMAL)
cv2.imshow('Canny Edges', canny_edges)
cv2.waitKey()

cv2.destroyAllWindows()

import cv2
import numpy as np

img = cv2.imread('retina_3.jpeg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
desiredResizedDimension = (img.shape[1], img.shape[0])
kernel = np.ones((3, 3), np.uint8)
img = cv2.resize(img, desiredResizedDimension)

directory = r'/home/adil/PycharmProjects/ImageProcessing/thesisPreparation'
filename = 'Gradient Gamma.jpeg'

# sobel
sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow('Original', img)
cv2.resizeWindow("Original", cols, rows)
cv2.waitKey()

cv2.namedWindow("Sobel Horizontal Filter", cv2.WINDOW_NORMAL)
cv2.imshow('Sobel Horizontal Filter', sobel_horizontal)
cv2.resizeWindow("Sobel Horizontal Filter", cols, rows)
cv2.waitKey()

cv2.namedWindow("Sobel Vertical Filter", cv2.WINDOW_NORMAL)
cv2.imshow('Sobel Vertical Filter', sobel_vertical)
cv2.resizeWindow("Sobel Vertical Filter", cols, rows)
cv2.waitKey()

# laplacian
laplacian_edges = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
cv2.namedWindow("Laplacian Filter", cv2.WINDOW_NORMAL)
cv2.imshow('Laplacian Filter', laplacian_edges)
cv2.resizeWindow("Laplacian Filter", cols, rows)
cv2.waitKey()

# gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
rows_gradient, cols_gradient = gradient.shape
cv2.namedWindow("Gradient", cv2.WINDOW_NORMAL)
cv2.imshow('Gradient', gradient)
cv2.resizeWindow("Laplacian Filter", cols_gradient, rows_gradient)
cv2.imwrite(filename, gradient)
cv2.waitKey()

# canny
canny_edges = cv2.Canny(img, 10, 10, edges=None, apertureSize=5)
cv2.namedWindow("Canny Edges", cv2.WINDOW_NORMAL)
cv2.imshow('Canny Edges', canny_edges)
cv2.resizeWindow("Canny Edges", cols, rows)
cv2.waitKey()


# gradient-gamma
original_gamma = cv2.imread(filename)


def adjust_gamma(image, gamma1=1.0):
    inv_gamma = 1.0 / gamma1
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


for gamma in np.arange(0.0, 3.5, 0.5):
    if gamma == 1:
        continue

    gamma = gamma if gamma > 0 else 0.1
    adjusted = adjust_gamma(original_gamma, gamma1=gamma)
    cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.namedWindow("Images-gamma", cv2.WINDOW_NORMAL)
    cv2.imshow("Images-gamma", np.hstack([original_gamma, adjusted]))
    cv2.resizeWindow("Images-gamma", cols, rows)
    cv2.waitKey()

cv2.destroyAllWindows()

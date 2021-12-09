import cv2
import numpy as np

# load the original image
imageToUseName = 'a_Airplane.jpg'
originalImg = cv2.imread(imageToUseName)

#illustrate the results
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", originalImg)
cv2.resizeWindow("Original Image", 1280, 720)

# Create kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
# Sharpen image
sharpImg = cv2.filter2D(originalImg, -1, kernel)

#illustrate the sharpened image results
cv2.namedWindow("Sharpened Image", cv2.WINDOW_NORMAL)
cv2.imshow("Sharpened Image", sharpImg)
cv2.resizeWindow("Sharpened Image", 1280, 720)

#blur the sharpened image
blurSharpImg = cv2.GaussianBlur(sharpImg, (5, 5), 0)

#and illustrate the blurred sharpened image results
cv2.namedWindow("Blurred Sharpened Image", cv2.WINDOW_NORMAL)
cv2.imshow("Blurred Sharpened Image", blurSharpImg)
cv2.resizeWindow("Blurred Sharpened Image", 1280, 720)

cv2.waitKey()
cv2.destroyAllWindows()

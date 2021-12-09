import cv2 #this is the main library
import numpy as np

# load the original image
imageToUseName = '7-Bali-Resorts-RIMBA-1.jpg'
originalImg = cv2.imread(imageToUseName)

#and illustrate the results
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Original Image", originalImg)
cv2.resizeWindow("Original Image", 480, 360)

# Create kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
# Sharpen image
sharpImg = cv2.filter2D(originalImg, -1, kernel)

#and illustrate the sharpened image results
cv2.namedWindow("Sharpened Image", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Sharpened Image", sharpImg)
cv2.resizeWindow("Sharpened Image", 480, 360)

#blur the sharpened image
blurSharpImg = cv2.GaussianBlur(sharpImg, (5, 5), 0)

#and illustrate the blurred sharpened image results
cv2.namedWindow("Blurred Sharpened Image", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Blurred Sharpened Image", blurSharpImg)
cv2.resizeWindow("Blurred Sharpened Image", 480, 360)

cv2.waitKey()
cv2.destroyAllWindows()



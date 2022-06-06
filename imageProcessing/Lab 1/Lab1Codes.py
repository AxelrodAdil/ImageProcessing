import cv2

# load the original image
imageToUseName = 'a_Airplane.jpg'
originalImg = cv2.imread(imageToUseName)

#and illustrate the results
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", originalImg)
cv2.resizeWindow("Original Image", 1280, 720)

#now grayscale it
imgGrayScale = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)
cv2.imshow("Grayscale Original Image", imgGrayScale)
cv2.resizeWindow("Grayscale Image", 1280, 720)


cv2.waitKey()
cv2.destroyWindow("Grayscale Image")

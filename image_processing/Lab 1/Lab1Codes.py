import cv2 #this is the main library

# load the original image
imageToUseName = '7-Bali-Resorts-RIMBA-1.jpg'
originalImg = cv2.imread(imageToUseName)

#and illustrate the results
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Original Image", originalImg)
cv2.resizeWindow("Original Image", 480, 360)

#now grayscale it
imgGrayScale = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Grayscale Original Image", imgGrayScale)
cv2.resizeWindow("Grayscale Image", 480, 360)


cv2.waitKey()
cv2.destroyWindow("Grayscale Image")

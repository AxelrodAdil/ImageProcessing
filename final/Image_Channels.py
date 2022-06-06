import cv2
from matplotlib import pyplot as plt

#read the basic image
imageToUseRGB = cv2.imread('a_Kaznu.jpg')

#percent of original size
scale_percent = 60
width = int(imageToUseRGB.shape[1] * scale_percent / 100)
height = int(imageToUseRGB.shape[0] * scale_percent / 100)
desiredResizedDimension = (width, height)

imageToUseRGBResized = cv2.resize(imageToUseRGB, desiredResizedDimension)

cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", imageToUseRGBResized)
cv2.waitKey()
cv2.destroyWindow('Original Image')

#color channels (BGR)
blueChanImg, greenChanImg, redChanImg = cv2.split(imageToUseRGBResized)

#illustration
cv2.namedWindow("Blue channel", cv2.WINDOW_NORMAL)
cv2.imshow("Blue channel", blueChanImg)
cv2.waitKey()
cv2.destroyWindow('Blue channel')
cv2.namedWindow("Green channel", cv2.WINDOW_NORMAL)
cv2.imshow("Green channel", greenChanImg)
cv2.waitKey()
cv2.destroyWindow('Green channel')
cv2.namedWindow("Red channel", cv2.WINDOW_NORMAL)
cv2.imshow("Red channel", redChanImg)
cv2.waitKey()
cv2.destroyWindow('Red channel')


#histograms per channel
#pixel values range from 0 to 255
blueHist = cv2.calcHist([blueChanImg], [0], None, [256], [0, 256])
greenHist = cv2.calcHist([greenChanImg], [0], None, [256], [0, 256])
redHist = cv2.calcHist([redChanImg], [0], None, [256], [0, 256])

# now normalize them
blueHist = blueHist/sum(blueHist)
greenHist = greenHist/sum(greenHist)
redHist = redHist/sum(redHist)

plt.plot(blueHist, label='blue')
plt.plot(greenHist, label='green')
plt.plot(redHist, label='red')
plt.show()
plt.close()

#histogram equalization
equalBlueChanImg = cv2.equalizeHist(blueChanImg)
equalGreenChanImg = cv2.equalizeHist(greenChanImg)
equalRedChanImg = cv2.equalizeHist(redChanImg)

#illustration of equalized images
cv2.namedWindow("Blue channel equalized", cv2.WINDOW_NORMAL)
cv2.imshow("Blue channel equalized", equalBlueChanImg)
cv2.waitKey()
cv2.destroyWindow('Blue channel equalized')
cv2.namedWindow("Green channel equalized", cv2.WINDOW_NORMAL)
cv2.imshow("Green channel equalized", equalGreenChanImg)
cv2.waitKey()
cv2.destroyWindow('Green channel equalized')
cv2.namedWindow("Red channel equalized", cv2.WINDOW_NORMAL)
cv2.imshow("Red channel equalized", equalRedChanImg)
cv2.waitKey()
cv2.destroyWindow('Red channel equalized')

#synthesize the gray images back to RGB
synthesizedRGB = cv2.merge((equalBlueChanImg, equalGreenChanImg, equalRedChanImg))

#finaly plot the image
cv2.namedWindow("Equalized Image", cv2.WINDOW_NORMAL)
cv2.imshow("Equalized Image", synthesizedRGB)
cv2.waitKey()
cv2.destroyWindow('Equalized Image')

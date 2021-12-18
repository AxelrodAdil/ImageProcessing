import cv2
from matplotlib import pyplot as plt


#read the basic image
imageToUseRGB = cv2.imread('redPanda.jpg')

#resize the image to demonstrate the functionality

scale_percent = 60 # percent of original size
width = int(imageToUseRGB.shape[1] * scale_percent / 100)
height = int(imageToUseRGB.shape[0] * scale_percent / 100)
desiredResizedDimension = (width, height)

imageToUseRGBResized = cv2.resize(imageToUseRGB, desiredResizedDimension)

cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Original Image", imageToUseRGBResized)
cv2.resizeWindow("Original Image", 480, 360)
cv2.waitKey(0)
cv2.destroyWindow('Original Image')

#now start investigating the color channels
# more details: https://stackabuse.com/introduction-to-image-processing-in-python-with-opencv/
blueChanIntenImg, greenChanIntenImg, redChanIntenImg = cv2.split(imageToUseRGBResized) # Split the image into its channels. CAREFULL it's BGR in OpenCV!

#and illustrate them!
cv2.namedWindow("Blue channel", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Blue channel", blueChanIntenImg)
cv2.resizeWindow("Blue channel", 480, 360)
cv2.waitKey(0)
cv2.namedWindow("Green channel", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Green channel", greenChanIntenImg)
cv2.resizeWindow("Green channel", 480, 360)
cv2.waitKey(0)
cv2.namedWindow("Red channel", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Red channel", redChanIntenImg)
cv2.resizeWindow("Red channel", 480, 360)
cv2.waitKey(0)


#create some histograms per channel
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html
#recal that pixel values range from 0 to 255. i.e. you will need 256 values to plot a full histogram
blueHist = cv2.calcHist([blueChanIntenImg], [0], None, [256], [0, 256])
greenHist = cv2.calcHist([greenChanIntenImg], [0], None, [256], [0, 256])
redHist = cv2.calcHist([redChanIntenImg], [0], None, [256], [0, 256])

# now normalize them
blueHist = blueHist/sum(blueHist)
greenHist = greenHist/sum(greenHist)
redHist = redHist/sum(redHist)

plt.plot(blueHist, label='blue')
plt.plot(greenHist, label='green')
plt.plot(redHist, label='red')
plt.show()                                                  #we need this to illustrate the plot
plt.waitforbuttonpress()
plt.close()


#destroy all windows
cv2.destroyAllWindows()

#attempt histogram equalization
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#:~:text=OpenCV%20has%20a%20function%20to%20do%20this%2C%20cv2.&text=Histogram%20equalization%20is%20good%20when,and%20dark%20pixels%20are%20present.
equalBlueChanIntenImg = cv2.equalizeHist(blueChanIntenImg)
equalGreenChanIntenImg = cv2.equalizeHist(greenChanIntenImg)
equalRedChanIntenImg = cv2.equalizeHist(redChanIntenImg)

#illustrate equalized images
#and illustrate them!
cv2.namedWindow("Blue channel equalized", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Blue channel equalized", equalBlueChanIntenImg)
cv2.resizeWindow("Blue channel equalized", 480, 360)
cv2.waitKey(0)
cv2.namedWindow("Green channel equalized", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Green channel equalized", equalGreenChanIntenImg)
cv2.resizeWindow("Green channel equalized", 480, 360)
cv2.waitKey(0)
cv2.namedWindow("Red channel equalized", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Red channel equalized", equalRedChanIntenImg)
cv2.resizeWindow("Red channel equalized", 480, 360)
cv2.waitKey(0)

#destroy all windows
cv2.destroyAllWindows()

#synthesize the gray images back to RGBsky
synthesizedRGB = cv2.merge((equalBlueChanIntenImg, equalGreenChanIntenImg, equalRedChanIntenImg))

#finaly plot the image
cv2.namedWindow("Equalized Image", cv2.WINDOW_NORMAL) # this allows for resizing using mouse
cv2.imshow("Equalized Image", synthesizedRGB )
cv2.resizeWindow("Equalized Image", 480, 360)
cv2.waitKey(0)
cv2.destroyWindow('Equalized Image')
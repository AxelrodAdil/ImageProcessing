#the opencv library
import cv2 #version 3.4.2
import numpy as np
import random


def sp_noise (image, prob):
    #add salt and pepper noise
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn >thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

#define image name/path
fileNameToUse = 'C:/Users/eftpr/Dropbox/PythonExamples/image processing/Lab 1/7-Bali-Resorts-RIMBA-1.jpg'

#read the image
originalImg = cv2.imread(fileNameToUse, 0)
originalImg.shape

#print the image
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL) #to resize at will
cv2.imshow("Original Image", originalImg)
cv2.waitKey()
cv2.destroyWindow("Original Image")

#create noisy image
noisyImage = sp_noise(originalImg, 0.2)

#print the image
cv2.namedWindow("Noisy Image", cv2.WINDOW_NORMAL) #to resize at will
cv2.imshow("Noisy Image", noisyImage)
cv2.waitKey()
cv2.destroyWindow("Noisy Image")
cv2.waitKey()
cv2.destroyWindow("Noisy Image")

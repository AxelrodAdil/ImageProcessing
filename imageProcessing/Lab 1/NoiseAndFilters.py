import cv2
import numpy as np
import random
import time
from skimage.metrics import structural_similarity as compare_ssim

#define function to create some noise to an image
def sp_noise(image, prob):
    """
    https://stackoverflow.com/questions/60056966/iterate-over-two-images-pixel-by-pixel-in-numpy-with-a-random-condition
    Add salt and pepper noise to image. Replaces random pixels with 0 or 1.
    prob: Probability of the noise
    """

    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def gs_noise(image):
    #gauss noise Gaussian-distributed additive noise.
    row, col, ch = image.shape
    mean = 0
    var = 0.4
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy

# #poisson Poisson-distributed noise generated from the data
# vals = len(np.unique(image))
# vals = 2 ** np.ceil(np.log2(vals))
# noisy = np.random.poisson(image * vals) / float(vals)
# return noisy

#load the image
img_BGR = cv2.imread('a_Bali.jpg')
#img_BGR = cv2.imread('a_Airplane.jpg')
#img_BGR = cv2.imread('a_Kaznu.jpg')
#img_BGR = cv2.imread('a_Dark_forest.jpg')

img_GRAY = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)

#and illustrate the results
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", img_BGR)
cv2.resizeWindow("Original Image", 1280, 720)

cv2.waitKey()
cv2.destroyWindow("Original Image")

cv2.namedWindow("Grayscale Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Grayscale Original Image", img_GRAY)
cv2.resizeWindow("Grayscale Original Image", 1280, 720)

cv2.waitKey()
cv2.destroyWindow("Grayscale Original Image")

#some filters and compare change to the original
print('Illustrating the effects of various filters, in terms of time and information loss')

#a. averaging filter
kernel = np.ones((5, 5), np.float32)/25
#averaging kernel for 5 x 5 window patch
print(kernel)

t = time.time()
blur = cv2.filter2D(img_BGR, -1, kernel)
tmpRunTime = time.time() - t

#illustrate the results
cv2.namedWindow("Averaging Filter", cv2.WINDOW_NORMAL)
cv2.imshow("Averaging Filter", blur)
cv2.resizeWindow("Averaging Filter", 1280, 720)

cv2.waitKey()
cv2.destroyWindow("Averaging Filter")

#calculate the similarity between the images
#compute the Structural Similarity Index (SSIM) between the two images
blur_GRAY = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

#illustrate the results
cv2.namedWindow("Averaging Filter Gray", cv2.WINDOW_NORMAL)
cv2.imshow("Averaging Filter Gray", blur_GRAY)
cv2.resizeWindow("Averaging Filter Gray", 1280, 720)

cv2.waitKey()
cv2.destroyWindow("Averaging Filter Gray")

#-----------------------------------------------------------------------------------------------------------

(score, diff) = compare_ssim(img_GRAY, blur_GRAY, full=True)
print(" .. Averaging blur filter time was: {:.8f}".format(tmpRunTime), " seconds. SSIM score: {:.4f}".format(score))

cv2.waitKey()

#b. gaussian filter
t = time.time()
blur = cv2.GaussianBlur(img_BGR, (5, 5), 0)
tmpRunTime = time.time() - t

#illustrate the results
cv2.namedWindow("Gauss blur Filter", cv2.WINDOW_NORMAL)
cv2.imshow("Gauss blur Filter", blur)
cv2.resizeWindow("Gauss blur Filter", 1280, 720)

cv2.waitKey()
cv2.destroyWindow("Gauss blur Filter")

#calculate the similarity between the images
#compute the Structural Similarity Index (SSIM) between the two images
blur_GRAY = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
(score, _) = compare_ssim(img_GRAY, blur_GRAY, full=True)
print(" .. Gaussian blur filter time was: {:.8f}".format(tmpRunTime), " seconds. SSIM score: {:.4f}".format(score))

cv2.waitKey()

#c. bilateral filter
t = time.time()
blur = cv2.bilateralFilter(img_BGR, 9, 5, 5)
tmpRunTime = time.time() - t

#illustrate the results
cv2.namedWindow("Bilateral Filter", cv2.WINDOW_NORMAL)
cv2.imshow("Bilateral Filter", blur)
cv2.resizeWindow("Bilateral Filter", 1280, 720)

cv2.waitKey()
cv2.destroyWindow("Bilateral Filter")

#calculate the similarity between the images
# compute the Structural Similarity Index (SSIM) between the two images
blur_GRAY = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
(score, _) = compare_ssim(img_GRAY, blur_GRAY, full=True)
print(" .. Bilateral blur filter time was: {:.8f}".format(tmpRunTime), " seconds. SSIM score: {:.4f}".format(score))

cv2.waitKey()

#distroy specific filter windows
#cv2.destroyWindow("Averaging Filter")
#cv2.destroyWindow("Gauss blur Filter")
#cv2.destroyWindow("Bilateral Filter")
#cv2.destroyAllWindows()

#create the new noisy image using salt and pepper
img_NOISY_BGR = sp_noise(img_BGR, 0.13)
#create the new noisy image using gauss noise
#img_NOISY_BGR = gs_noise(img_BGR)
img_NOISY_BGR = np.floor(np.abs(img_NOISY_BGR)).astype('uint8')

img_NOISY_GRAY = cv2.cvtColor(img_NOISY_BGR, cv2.COLOR_BGR2GRAY)

#illustrate the results
cv2.namedWindow("Original (noisy) Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original (noisy) Image", img_NOISY_BGR)
cv2.resizeWindow("Original (noisy) Image", 1280, 720)

cv2.waitKey()
cv2.destroyWindow("Original (noisy) Image")

cv2.namedWindow("Grayscale Original (noisy) Image", cv2.WINDOW_NORMAL)
cv2.imshow("Grayscale Original (noisy) Image", img_NOISY_GRAY)
cv2.resizeWindow("Grayscale Original (noisy) Image", 1280, 720)

cv2.waitKey()
cv2.destroyWindow("Grayscale Original (noisy) Image")

print("Demonstrating the noise reduction capabilities for each of the filters")

(score, _) = compare_ssim(img_GRAY, img_NOISY_GRAY, full=True)
print(" .. Currently SSIM score between original and noise image is: {:.4f}".format(score))
print(" .. Attempting to fix results, using different filters")

cv2.waitKey()

#some filters and compare change to the original
print('Illustrating the effects of various filters, in terms of time and information loss')
print("\n---------------------------\n197-LINE\n---------------------------\n")

#-----------------------------------------------------------------------------------------------------------

#a. averaging filter
kernel = np.ones((5, 5), np.float32) / 25
print(kernel)

t = time.time()
blur = cv2.filter2D(img_NOISY_BGR, -1, kernel)
tmpRunTime = time.time() - t

#illustrate the results
cv2.namedWindow("Averaging Filter", cv2.WINDOW_NORMAL)
cv2.imshow("Averaging Filter", blur)
cv2.resizeWindow("Averaging Filter", 1280, 720)

cv2.waitKey()
cv2.destroyWindow("Averaging Filter")

#calculate the similarity between the images
# compute the Structural Similarity Index (SSIM) between the two images
blur_GRAY = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
(score, _) = compare_ssim(img_GRAY, blur_GRAY, full=True)
print(" .. Averaging blur filter time was: {:.8f}".format(tmpRunTime), " seconds. SSIM score: {:.4f}".format(score))

cv2.waitKey()

#b. gaussian filter
t = time.time()
blur = cv2.GaussianBlur(img_NOISY_BGR, (5, 5), 0)
tmpRunTime = time.time() - t

#illustrate the results
cv2.namedWindow("Gauss blur Filter", cv2.WINDOW_NORMAL)
cv2.imshow("Gauss blur Filter", blur)
cv2.resizeWindow("Gauss blur Filter", 1280, 720)

cv2.waitKey()
cv2.destroyWindow("Gauss blur Filter")

#calculate the similarity between the images
# compute the Structural Similarity Index (SSIM) between the two images
blur_GRAY = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
(score, _) = compare_ssim(img_GRAY, blur_GRAY, full=True)
print(" .. Gaussian blur filter time was: {:.8f}".format(tmpRunTime), " seconds. SSIM score: {:.4f}".format(score))

cv2.waitKey()

#c. bilateral filter
t = time.time()
blur = cv2.bilateralFilter(img_NOISY_BGR, 9, 5, 5)
tmpRunTime = time.time() - t

#illustrate the results
cv2.namedWindow("Bilateral Filter", cv2.WINDOW_NORMAL)
cv2.imshow("Bilateral Filter", blur)
cv2.resizeWindow("Bilateral Filter", 1280, 720)

cv2.waitKey()
cv2.destroyWindow("Bilateral Filter")

#calculate the similarity between the images
# compute the Structural Similarity Index (SSIM) between the two images
blur_GRAY = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
(score, _) = compare_ssim(img_GRAY, blur_GRAY, full=True)
print(" .. Bilateral blur filter time was: {:.8f}".format(tmpRunTime), " seconds. SSIM score: {:.4f}".format(score))

cv2.waitKey()

#distroy specific filter windows
print("\n-END-")
cv2.destroyAllWindows()

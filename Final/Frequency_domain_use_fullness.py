import cv2
import numpy as np
from matplotlib import pyplot as plt

#read an image in grayscale
image = cv2.imread('a_Kaznu.jpg', 0)
rows, cols = image.shape
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow('Original', image)
cv2.waitKey()

# convert from uint8 into float32
image_float32 = np.float32(image)

# Computed the 2-d discrete Fourier Transform
dft = cv2.dft(image_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
# Shift the zero-frequency component to the center of the spectrum.
dft_shift = np.fft.fftshift(dft)
# compute magnitude spectrum
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

#a low pass filtering; i.e cut all frequencies above a specific threshold
r = 50
#1D hamming
ham = np.hamming(min(rows, cols))[:, None]
#expand to 2D hamming
ham2d = np.sqrt(np.dot(ham, ham.T)) ** r
ham2d = cv2.resize(ham2d, (cols, rows), interpolation=cv2.INTER_AREA)

#plot the filter
#https://www.l3harrisgeospatial.com/docs/lowpassfilter.html
cv2.namedWindow("LPF", cv2.WINDOW_NORMAL)
cv2.imshow('LPF', ham2d)
cv2.waitKey()

# apply mask
dft_shift[:, :, 0] *= ham2d
dft_shift[:, :, 1] *= ham2d

#run an inverse fourier
inverse_Fourier = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(inverse_Fourier)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

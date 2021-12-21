#demonstrating the usefullness of frequency domain
import cv2
import numpy as np
from matplotlib import pyplot as plt

#read an image in grayscale

image = cv2.imread('crazy_road.jpg', 0)

#we will need these two latter
rows, cols = image.shape

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow('Original', image)
cv2.waitKey()

cv2.waitKey()
cv2.destroyWindow('Original')

image_float32 = np.float32(image) # convert from uint8 into float32
dft = cv2.dft(image_float32, flags = cv2.DFT_COMPLEX_OUTPUT) # Computed the 2-d discrete Fourier Transform
dft_shift = np.fft.fftshift(dft) # Shift the zero-frequency component to the center of the spectrum.
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])) # compute magnitude spectrum


plt.subplot(121),plt.imshow(image, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# cv2.destroyAllWindows()

#now try a low pass filtering; i.e cut all frequencies above a specific threshold

#start by creating a hamming window (a.k.a. the filter)
r = 50 # how narrower the window is
ham = np.hamming(min(rows,cols))[:,None] # 1D hamming
ham2d = np.sqrt(np.dot(ham, ham.T)) ** r # expand to 2D hamming

#make sure that is has the same size as the input image
# ham2d = np.resize(ham2d, (rows, cols))
#ham2d = cv2.copyMakeBorder(ham2d,)
ham2d = cv2.resize(ham2d, (cols, rows), interpolation=cv2.INTER_AREA)

#plot the filter
cv2.namedWindow("LPF", cv2.WINDOW_NORMAL)
cv2.imshow('LPF', ham2d)
cv2.waitKey()

cv2.destroyWindow('LPF')

# apply mask (yes, it's just a multiplication)
#fshift = dft_shift * ham2d #this causes an error due to dimensions
# dft_complex = dft_shift[:,:,0]*1j + dft_shift[:,:,1]
# dft_shift = ham2d * dft_complex
dft_shift[:,:,0] *= ham2d
dft_shift[:,:,1] *= ham2d

#run an inverse fourrier
f_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(image, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
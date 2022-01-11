from __future__ import print_function
import numpy as np
import cv2

def adjust_gamma(image, gamma1=1.0):
    inv_gamma = 1.0 / gamma1
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# load the original image
imageToUseName = 'a_Kaznu.jpg'

original = cv2.imread(imageToUseName)

# loop over various values of gamma
for gamma in np.arange(0.0, 3.5, 0.5):
    # ignore when gamma is 1 (there will be no change to the image)
    if gamma == 1:
        continue

    # apply gamma correction and show the images
    gamma = gamma if gamma > 0 else 0.1
    adjusted = adjust_gamma(original, gamma1=gamma)
    cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
    cv2.imshow("Images", np.hstack([original, adjusted]))
    cv2.waitKey()

cv2.destroyAllWindows()

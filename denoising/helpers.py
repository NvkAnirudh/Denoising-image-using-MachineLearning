import numpy as np
import cv2

def blur_and_threshold(image, eps=1e-7):
    # 1) add median blur to input image
    # 2) Subtract the median blur from original image
    blur = cv2.medianBlur(image, 5)
    foreground = image.astype('float') - blur

    # 3) threshold the subtracted image by setting any pixels with a value greater than zero to zero
    foreground[foreground > 0] = 0

    # 4) Apply min-max scaling to bring the pixel intensities to [0,1]
    minVal = np.min(foreground)
    maxVal = np.max(foreground)
    foreground = (foreground - minVal) / (maxVal - minVal + eps)

    return foreground
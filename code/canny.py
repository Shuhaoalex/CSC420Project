import numpy as np
import cv2 as cv

def canny(img, guassian_blur_sigma=1.0, sobel_size=3, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    """
    A simple wrapper for OpenCV's Canny function, using a dynamic thresholding strategy instead of a fixed value one
    """
    g_ksize = int(guassian_blur_sigma * 6) - 1
    blurred = cv.GaussianBlur(img, (g_ksize, g_ksize), guassian_blur_sigma)
    gx = cv.Sobel(cv.cvtColor(blurred, cv.COLOR_BGR2GRAY), cv.CV_16SC1, 1, 0, ksize=sobel_size)
    gy = cv.Sobel(cv.cvtColor(blurred, cv.COLOR_BGR2GRAY), cv.CV_16SC1, 0, 1, ksize=sobel_size)
    g = np.hypot(gx, gy).max()
    highThreshold = g * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    return cv.Canny(gx, gy, lowThreshold, highThreshold, L2gradient=True)
import cv2
import numpy as np
from PIL import Image, ImageFilter


def single_scale_retinex(img, sigma):

    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)

    # Avoid division by zero
    blurred = np.where(blurred == 0, 1, blurred)

    # Compute the Retinex
    retinex = np.log1p(img) - np.log1p(blurred)
    return retinex

def multi_scale_retinex(img, sigma_list):




    retinex = np.zeros_like(img, dtype=np.float32)

    for sigma in sigma_list:
        retinex += single_scale_retinex(img, sigma)

    # Average across the scales
    retinex = retinex / len(sigma_list)

    return retinex

def retinex(image_path, output, sigma_list=[15,45, 70,  80, 120, 150, 180,  200]):


    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert to the LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply Multi-Scale Retinex to the L (lightness) channel
    l = l.astype(np.float32) / 255  # Normalize to [0, 1]
    retinex_result = multi_scale_retinex(l, sigma_list)
    retinex_result = cv2.normalize(retinex_result, None, 0, 255, cv2.NORM_MINMAX)
    l = retinex_result.astype(np.uint8)

    # Merge back with A and B channels
    enhanced_lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    cv2.imwrite(output, enhanced_image)

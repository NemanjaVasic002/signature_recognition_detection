
from PIL import Image, ImageFilter
import numpy as np

#create image from list
def l2i(list, nameFile):
    mask2 = np.array(list)

    mask2 = list.astype(np.uint8)


    rgb_mask2 = np.stack([mask2] * 3, axis=-1)

    image3 = Image.fromarray(rgb_mask2)

    image3 = image3.save(nameFile)
    return image3

#create list from image
def lti(mask_list):
    mask_array = np.array(mask_list).astype(np.uint8)

    # Stack to create RGB image (3 channels)
    rgb_mask = np.stack([mask_array] * 3, axis=-1)

    return rgb_mask  # Return NumPy array instead of Image

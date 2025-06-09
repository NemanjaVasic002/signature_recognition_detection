import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2000000000
import os


from signature_detect.loader import Loader
from signature_detect.extractor import Extractor
from signature_detect.cropper import Cropper
from signature_detect.judger import Judger

from imageFromlist import l2i
from imageFromlist import lti


from retinex import  retinex



def signatureDetection(imagePath,saveDir,signatureNumber,imgPro):
#imagePath path to the image from which you want to extract siganture
#saveDir directory where you want to save your signatue
#number of your signature
#imgpro this is directory where you save img in processing period so you can see whats wrong



    os.makedirs(saveDir, exist_ok=True)  # Make sure directory exists
    os.makedirs(imgPro, exist_ok=True)  # Make sure directory exists
#if loader isnt taking your whole signature use histogram.py, and change only last number in low_threshold aka value
#if cropper isnt giving you your signature as whole set low_threshold little higher, still before max value of the hist
#for extractor this is best solution for image shot with smartphone camera, for scanned img you should use different values
    loader = Loader(
        low_threshold=(0,0,180),
        high_threshold=(255, 255, 255))
    extractor = Extractor(
        outlier_weight=1,
        outlier_bias=100,
        amplfier=25,
        min_area_size=1000)
    #def 1000, 0.01
    cropper = Cropper(
        min_region_size=1000,
        border_ratio=0.01)
    # def 1, 4 0,01 1
    judger = Judger(
        size_ratio=[1, 4],
        pixel_ratio=[0.1, 1])







    exitPath = os.path.join(imgPro, f"Signature.jpg")

    #with retinex filter we will reduce shadow effect
    retinex(imagePath, exitPath)


    mask = loader.get_masks(exitPath)[0]
    exitPath = os.path.join(imgPro, f"loader.jpg")
    cv2.imwrite(exitPath,mask)

    labeled_mask = extractor.extract(mask)
    exitPath = os.path.join(imgPro, f"extractorMask.jpg")
    image2 = l2i(labeled_mask, exitPath)


    results = cropper.run(labeled_mask)
    #if you want to see size of the regions uncomment
    # num_regions = len(results)
    # print(f"Number of regions cropped: {num_regions}")
    # total_area = 0
    # total_width = 0
    # total_height = 0

    # for key, region_info in results.items():
    #     x, y, w, h = region_info["cropped_region"]
    #     total_area += w * h
    #     total_width += w
    #     total_height += h
    #
    # avg_area = total_area / num_regions if num_regions else 0
    # avg_width = total_width / num_regions if num_regions else 0
    # avg_height = total_height / num_regions if num_regions else 0
    #
    # print(f"Average area of regions: {avg_area:.2f} pixels")
    # print(f"Average width of regions: {avg_width:.2f} pixels")
    # print(f"Average height of regions: {avg_height:.2f} pixels")




    number = 0


    output = judger.run(results)

    number = 0
    counter = 0
    for result in results.values():
        number = result["cropped_mask"]
        is_signed = judger.judge(number)

        if is_signed:
            counter += 1

            if lti(number).size <= 10000:
                break

            imagePath = os.path.join(saveDir, f"signature{signatureNumber}{counter}.jpg")
            image5 = l2i(number, imagePath)
            image2 = number
            if counter == 2:
                image1 = lti(number)


                if np.array_equal(image1, image2):
                    counter = 1
                else:
                    imagePath = os.path.join(saveDir, f"signature{signatureNumber}{counter}.jpg")
                    image5 = l2i(number, imagePath)
                    break




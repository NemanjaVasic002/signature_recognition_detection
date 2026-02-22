import cv2

path = r"D:\SignatureDetection\ProjekatSOS\verification\training5\PNV6 1.jpg"
image = cv2.imread(path)


image = cv2.resize(image, (1124, 512), interpolation=cv2.INTER_AREA)


cv2.imwrite(path, image)
import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class SignatureDataset(Dataset):
    def __init__(self, csvFile, rootDirectory, imageTransform=None):
        self.dataFrame = pd.read_csv(csvFile)
        self.rootDirectory = rootDirectory
        self.imageTransform = imageTransform

    def __len__(self):
        return len(self.dataFrame)

    def loadImage(self, fileName):
        imagePath = os.path.join(self.rootDirectory, fileName + '.jpg')
        loadedImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        if loadedImage is None:
            raise FileNotFoundError(f"Image not found: {imagePath}")

        if self.imageTransform:
            loadedImage = self.imageTransform(loadedImage)
        return loadedImage

    def __getitem__(self, index):
        anchorName = self.dataFrame.iloc[index, 0]
        positiveName = self.dataFrame.iloc[index, 1]
        negativeName = self.dataFrame.iloc[index, 2]

        anchorImg = self.loadImage(anchorName)
        positiveImg = self.loadImage(positiveName)
        negativeImg = self.loadImage(negativeName)

        return anchorImg, positiveImg, negativeImg
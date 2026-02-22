import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseSignatureNet(nn.Module):
    def __init__(self):
        super(SiameseSignatureNet, self).__init__()

        self.featureLayers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        with torch.no_grad():
            dummyInput = torch.zeros(1, 1, 512, 512)
            dummyOutput = self.featureLayers(dummyInput)
            self.flattenedSize = dummyOutput.numel()

        self.fullyConnected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattenedSize, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.outputLayer = nn.Linear(256, 1)

    def forwardOnce(self, x):
        x = self.featureLayers(x)
        x = self.fullyConnected(x)
        return x

    def forward(self, inputOne, inputTwo):
        outputOne = self.forwardOnce(inputOne)
        outputTwo = self.forwardOnce(inputTwo)
        distanceVector = torch.abs(outputOne - outputTwo)

        return torch.sigmoid(self.outputLayer(distanceVector))
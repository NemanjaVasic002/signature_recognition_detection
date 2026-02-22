import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletSignatureNet(nn.Module):
    def __init__(self):
        super(TripletSignatureNet, self).__init__()

        self.featureLayers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        with torch.no_grad():
            dummyInput = torch.zeros(1, 1, 512, 512)
            dummyOutput = self.featureLayers(dummyInput)
            self.flattenedSize = dummyOutput.numel()

        self.fullyConnected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattenedSize, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64)
        )

    def forwardOnce(self, x):
        x = self.featureLayers(x)
        x = self.fullyConnected(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, anchor, positive, negative):
        return self.forwardOnce(anchor), self.forwardOnce(positive), self.forwardOnce(negative)
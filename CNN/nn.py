import torch
import torch.nn as nn
import torch.nn.functional as F


class SignatureCNN(nn.Module):
    def __init__(self):
        super(SignatureCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # 8x512x512
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x256x256

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # 16x256x256
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x128x128

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 32x128x128
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x64x64

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64x64x64
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x32x32
        )

        # Calculate output feature size dynamically

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 512, 1124)  # adjust if you change input size
            out = self.features(dummy_input)
            self.flattened_size = out.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
    )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

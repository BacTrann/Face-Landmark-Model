import torch.nn as nn
from torchvision import models


class FaceLandModel(nn.Module):
    # Build from pretrained network resnet18
    def __init__(self, num_classes=136):
        super().__init__()
        self.model_name = 'resnet18'
        self.model = models.resnet18()

        # Input channel: 1 to accept grayscale images
        # Output channel: 68(num landmarks) * 2(x,y) = 136 
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

"""Classifier for the extraction of features."""

from torch import nn


class MNISTClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2) # (1, 28, 28) -> (6, 28, 28)
        self.ReLU1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2) # (6, 28, 28) -> (6, 14, 14)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0) # (6, 14, 14) -> (16, 10, 10)
        self.ReLU2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=2) # (16, 10, 10) -> (16, 5, 5)
        self.flatten = nn.Flatten() # (16, 5, 5) -> (400,)
        self.fc1 = nn.Linear(400, 120) # (400,) -> (120,)
        self.ReLU3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84) # (120,) -> (84,)
        self.ReLU4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10) # (84,) -> (10,)

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.ReLU1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.ReLU2(x)
        x = self.avgpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ReLU3(x)
        x = self.fc2(x)
        x = self.ReLU4(x)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.fc3(x)
        return x



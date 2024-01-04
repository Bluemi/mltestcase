import torch
import torch.nn as nn
import torch.nn.functional as functional


class MnistClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.pool2(functional.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MnistAutoencoder(nn.Module):
    def __init__(self):
        bottleneck = 2
        middle = 100
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, middle)
        self.fc2 = nn.Linear(middle, bottleneck)
        self.fc3 = nn.Linear(bottleneck, middle)
        self.fc4 = nn.Linear(middle, 28 * 28)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = torch.flatten(x, start_dim=1)
        x = functional.elu(self.fc1(x))
        return self.fc2(x)

    def decode(self, x):
        x = functional.elu(self.fc3(x))
        x = self.fc4(x)
        return x

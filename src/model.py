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


def noop(x):
    return x


class MnistAutoencoder(nn.Module):
    def __init__(self, activation_func: str = 'sigmoid', use_activation_for_z=False, training=False):
        super().__init__()
        self.training = training

        bottleneck = 2
        middle = 100

        if activation_func == 'sigmoid':
            activation_function = nn.Sigmoid
        elif activation_func == 'tanh':
            activation_function = nn.Tanh
        elif activation_func == 'relu':
            activation_function = nn.ReLU
        elif activation_func == 'elu':
            activation_function = nn.ELU
        else:
            raise ValueError('Unknown activation function: {}'.format(activation_func))

        encoder_layers = [
            nn.Linear(28 * 28, middle),
            activation_function(),
            nn.Linear(middle, bottleneck)
        ]
        if use_activation_for_z:
            encoder_layers.append(activation_function())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [
            nn.Linear(bottleneck, middle),
            activation_function(),
            nn.Linear(middle, 28 * 28)
        ]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = torch.flatten(x, start_dim=1)

        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

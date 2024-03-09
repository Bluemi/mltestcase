import torch
from torch import nn as nn
from torch.nn import functional as functional

from model.layers import BlobLayer, MothLayer
from utils import describe


class MnistAutoencoder(nn.Module):
    def __init__(self, activation_func: str = 'sigmoid', use_activation_for_z=False, use_blob_layer=False, training=False):
        super().__init__()
        self.training = training

        bottleneck = 2
        middle = 100

        if activation_func == 'sigmoid':
            activation_function = lambda n: nn.Sigmoid()
        elif activation_func == 'tanh':
            activation_function = lambda n: nn.Tanh()
        elif activation_func == 'relu':
            activation_function = lambda n: nn.ReLU()
        elif activation_func == 'elu':
            activation_function = lambda n: nn.ELU()
        elif activation_func == 'moth':
            activation_function = MothLayer
        else:
            raise ValueError('Unknown activation function: {}'.format(activation_func))

        if use_blob_layer:
            first_layer = BlobLayer(middle, image_size=(28, 28))
        else:
            first_layer = nn.Linear(28 * 28, middle)

        encoder_layers = [
            first_layer,
            nn.Sigmoid(),
            activation_function(middle),
            nn.Linear(middle, bottleneck)
        ]
        if use_activation_for_z:
            encoder_layers.append(nn.Sigmoid())
            encoder_layers.append(activation_function(bottleneck))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [
            nn.Linear(bottleneck, middle),
            nn.Sigmoid(),
            activation_function(middle),
            nn.Linear(middle, 28 * 28)
        ]
        self.decoder = nn.Sequential(*decoder_layers)

        classification_layers = [
            nn.Linear(bottleneck, 10),
        ]
        self.classification_head = nn.Sequential(*classification_layers)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def forward_classify(self, x):
        x = self.encode(x)
        return self.classification_head(x)

    def encode(self, x):
        x = torch.flatten(x, start_dim=1)

        # print()
        # describe(x, 'x')
        # for p in self.encoder[0].parameters():
        #     describe(p, 'p')
        # if torch.isnan(x).any():
        #     raise ValueError('x isnan before first layer')
        # x = self.encoder[0](x)
        # describe(x, 'x after first layer')
        # x = self.encoder[1](x)
        # describe(x, 'x after second layer')
        # x = self.encoder[2](x)
        # describe(x, 'x after third layer')

        # return x

        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


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

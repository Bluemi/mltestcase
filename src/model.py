import numpy as np
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


class CustomLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = None
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(self._in_features))
        if self.bias is not None:
            # noinspection PyProtectedMember
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.matmul(x, self.weights)
        if self.bias is not None:
            x = x + self.bias
        return x


class BlobLayer(nn.Module):
    def __init__(self, num_curves, image_size, tau=0.01):
        super().__init__()
        self.image_size = image_size
        self.tau = tau
        self.positions = nn.Parameter(torch.normal(0.5, 0.3, size=(num_curves, 2)))
        self.sigmas = nn.Parameter(torch.normal(0, 0.02, size=(num_curves,)))

        y_axis = torch.linspace(0, 1, image_size[0] + 1)[:-1]
        x_axis = torch.linspace(0, 1, image_size[1] + 1)[:-1]
        ys, xs = torch.meshgrid(y_axis, x_axis, indexing='ij')
        self.register_buffer('ys', ys, persistent=False)
        self.register_buffer('xs', xs, persistent=False)

    def calc_curves(self):
        factor = 1 / (2 * torch.pi * self.sigmas[None, None, :] ** 2 + self.tau)

        # shape of grid: (IMAGE_SIZE_Y, IMAGE_SIZE_X, N_CURVES)
        grid = (self.xs[:, :, None] - self.positions[None, None, :, 1]) ** 2 + (self.ys[:, :, None] - self.positions[None, None, :, 0]) ** 2
        second_part = torch.exp(- grid / (2 * self.sigmas[None, None, :] ** 2 + self.tau))
        return factor * second_part

    def forward(self, x):
        x = x.reshape(-1, *self.image_size)
        curves = self.calc_curves()
        prod = curves[None] * x[..., None]
        return torch.sum(prod, dim=(1, 2))

class MnistAutoencoder(nn.Module):
    def __init__(self, activation_func: str = 'sigmoid', use_activation_for_z=False, use_blob_layer=False, training=False):
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

        if use_blob_layer:
            first_layer = BlobLayer(middle, image_size=(28, 28))
        else:
            first_layer = nn.Linear(28 * 28, middle)

        encoder_layers = [
            first_layer,
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

        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

import numpy as np
import torch
from torch import nn as nn

from utils import describe


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
    def __init__(self, num_curves, image_size, epsilon=0.001, cap=2000):
        super().__init__()
        self.image_size = image_size
        self.num_pixels = float(np.prod(self.image_size))
        self.epsilon = epsilon
        self.cap = cap
        self.positions = nn.Parameter(torch.normal(0.5, 0.3, size=(1, 1, num_curves, 2)))
        self.sigmas = nn.Parameter(torch.normal(0, 0.02, size=(1, 1, num_curves)))
        self.curve_weights = nn.Parameter(torch.normal(0, 0.2, size=(1, 1, num_curves)))

        y_axis = torch.linspace(0, 1, image_size[0] + 1)[:-1]
        x_axis = torch.linspace(0, 1, image_size[1] + 1)[:-1]
        ys, xs = torch.meshgrid(y_axis, x_axis, indexing='ij')
        self.register_buffer('ys', ys, persistent=False)
        self.register_buffer('xs', xs, persistent=False)

    def calc_curves(self):
        factor = 1 / (2 * torch.pi * self.sigmas ** 2 + self.epsilon)
        # factor = torch.exp(-2.0 * self.sigmas + 1)  # alternative but worse

        # shape of grid: (IMAGE_SIZE_Y, IMAGE_SIZE_X, N_CURVES)
        grid = (self.xs[:, :, None] - self.positions[..., 1]) ** 2 + (self.ys[:, :, None] - self.positions[..., 0]) ** 2
        second_part = torch.exp(- grid / (2 * self.sigmas ** 2 + self.epsilon))
        curves = factor * second_part * self.curve_weights
        return torch.clamp(curves, -self.cap, self.cap)

    def forward(self, x):
        x = x.reshape(-1, *self.image_size)
        curves = self.calc_curves()
        prod = curves[None] * x[..., None]
        return torch.sum(prod, dim=(1, 2)) / self.num_pixels


class MothLayer(nn.Module):
    def __init__(self, num_features, bypass: bool = False):
        super().__init__()
        self.num_features = num_features
        self.bypass = bypass
        self.interpolation_factor = nn.Parameter(torch.normal(0, 0.2, size=(1, num_features)))
        # self.plane_gradient = nn.Parameter(torch.normal(1, 0.2, size=(1, 2, num_features)))

    def forward(self, x):
        x_limited = torch.tanh(0.2 * x)
        y = torch.roll(x_limited, shifts=-1, dims=-1)
        inter_fac = self.interpolation_factor + 0.5

        # a = (x*self.plane_gradient[:, 0] + y*self.plane_gradient[:, 1]) * (1 - inter_fac + 0.5)
        # a = x + y
        a = torch.sigmoid(x)

        # b = torch.abs(x - y) * (inter_fac + 0.5)
        def _q(w):
            return torch.minimum(torch.exp(w)-1, torch.maximum(w, torch.tensor(0.0)))
        b = _q(x_limited - y) + _q(y - x_limited)

        result = a * (1 - inter_fac) + b * inter_fac
        if self.bypass:
            return result + x_limited

        return result

from typing import Tuple, Union

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as functional


def conv2d_output_shape(input_size: Union[int, Tuple[int, int]], kernel_size, stride=1, padding=0, dilation=1):
    """
    input_size: numpy array or list/tuple of shape (H, W)
    kernel_size, stride, padding, dilation: int or tuple
    Returns: tuple (h_out, w_out)
    """
    def _pair(x):
        return (x, x) if isinstance(x, int) else x

    h_in, w_in = input_size
    k_h, k_w = _pair(kernel_size)
    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)
    d_h, d_w = _pair(dilation)

    h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
    w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1

    return h_out, w_out


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
        # x_limited = torch.tanh(0.2 * x)
        y = torch.roll(x, shifts=-1, dims=-1)
        inter_fac = (self.interpolation_factor + 0.5) * 0.2

        # a = (x*self.plane_gradient[:, 0] + y*self.plane_gradient[:, 1]) * (1 - inter_fac + 0.5)
        # a = x + y
        a = torch.sigmoid(x)

        # b = torch.abs(x - y) * (inter_fac + 0.5)
        # def _q(w):
        # return torch.minimum(torch.exp(w)-1, torch.maximum(w, torch.tensor(0.0)))
        def _v(w):
            s = torch.sigmoid(w)
            return w * s - w * (1-s)

        # b = _q(x - y) + _q(y - x)
        b = _v(x-y)

        result = a * (1 - inter_fac) + b * inter_fac
        if self.bypass:
            return result + x

        return result


class Conv2dMoth(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
            moth_channels: int | float = 0.2, moth_stride: int = 1, min_size: int = 10,
    ):
        super().__init__()
        self.conf = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if isinstance(moth_channels, float):
            moth_channels = int(moth_channels * out_channels)
        self.moth_channels = moth_channels
        self.moth_stride = moth_stride
        self.min_size = min_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate moth layer output.
        :param x: The tensor with shape [b, c, h, w].
        :return: Tensor with shape [b, c, h, w].
        """
        x = self.conf(x)
        if x.shape[2] >= self.min_size:
            d = torch.abs(
                x[:, :self.moth_channels, self.moth_stride*2:, :] -
                x[:, :self.moth_channels, :-self.moth_stride*2, :]
            )
            x[:, :self.moth_channels, self.moth_stride:-self.moth_stride, :] += d
        return x


class MothReLU2d(nn.Module):
    def __init__(self, abs_channels: int | float):
        super().__init__()
        self.abs_channels = abs_channels

    def forward(self, x: torch.Tensor):
        n, c, h, w = x.shape
        if isinstance(self.abs_channels, float):
            abs_channels = int(self.abs_channels * c)
        else:
            abs_channels = self.abs_channels
        abs_data = torch.abs(x[:, :abs_channels])
        relu_data = functional.relu(x[:, abs_channels:])
        x = torch.cat([abs_data, relu_data], dim=1)
        return x


class SuppressionLayer(nn.Module):
    def __init__(
            self, in_channels: int, input_size: Tuple[int, int], kernel_size: int = 1, stride: int = 1,
            padding: int = 0, reduction_features: int = 1
    ):
        super().__init__()
        self.reduction_features = reduction_features
        self.input_size = input_size
        self.num_output_features = int(np.prod(input_size))
        self.in_channels = in_channels
        self.linear_input_size = conv2d_output_shape(
            input_size, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.linear_num_input_features = int(np.prod(self.linear_input_size)) * self.reduction_features

        self.conv = nn.Conv2d(in_channels, reduction_features, kernel_size=kernel_size, stride=stride, padding=padding)
        self.linear = nn.Linear(self.linear_num_input_features, self.num_output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Input tensor with shape [b, c, h, w].
        :return:
        """
        batch_size = x.shape[0]
        mask = functional.sigmoid(self.conv(x))
        mask = mask.reshape(batch_size, self.linear_num_input_features)
        mask = functional.sigmoid(self.linear(mask))
        mask = mask.reshape(batch_size, 1, *self.input_size)

        x = x * mask
        return x


class ParameterizedLayer:
    def __init__(self, layer_type, **kwargs):
        self.layer_type = layer_type
        self.args = kwargs

    def __call__(self):
        return self.layer_type(**self.args)


class StatusLayer(nn.Module):
    def __init__(self, message: str = ''):
        super().__init__()
        self.message = message

    def forward(self, x):
        print(self.message, x.shape)
        return x

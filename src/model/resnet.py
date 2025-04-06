"""
Taken from https://d2l.ai/chapter_convolutional-modern/resnet.html
"""
from dataclasses import dataclass
from typing import List, Union, Tuple

from torch import nn

from model.layers import SuppressionLayer


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


class Residual(nn.Module):
    """The Residual block of ResNet models."""

    def __init__(
            self, in_channels: int, out_channels: int, input_size: Tuple[int, int], use_suppression: bool,
            use_1x1conv: bool = False, strides: int = 1, layer_type=nn.Conv2d, activation_type=nn.ReLU,
    ):
        super().__init__()
        self.input_size = input_size
        suppression_input_size = input_size
        self.in_channels = in_channels

        self.conv1 = layer_type(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        suppression_input_size = conv2d_output_shape(suppression_input_size, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.LazyBatchNorm2d()
        self.activation = activation_type()

        self.conv2 = layer_type(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.LazyBatchNorm2d()
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = layer_type(in_channels, out_channels, kernel_size=1, stride=strides)

        self.suppression_input_size = suppression_input_size
        if use_suppression:
            self.suppression = SuppressionLayer(out_channels, suppression_input_size, kernel_size=5, padding=2)
        else:
            self.suppression = None

    def forward(self, x):
        y = self.activation(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        if self.suppression:
            y = self.suppression(y)
        return self.activation(y)


@dataclass
class ArchBlock:
    num_residuals: int
    prev_num_channels: int
    num_channels: int


class ResNet(nn.Module):
    def __init__(
            self, input_size: Tuple[int, int], arch: List[ArchBlock], num_classes: int, use_suppression: bool,
            layer_type=nn.Conv2d, activation_type=nn.ReLU
    ):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(self.b1(3, 64, layer_type=layer_type, activation_type=activation_type))
        input_size = conv2d_output_shape(input_size, kernel_size=7, stride=2, padding=3)
        input_size = conv2d_output_shape(input_size, kernel_size=3, stride=2, padding=1)
        for i, b in enumerate(arch):
            self.net.add_module(
                f'b{i + 2}',
                self.block(
                    i, b, input_size, use_suppression=use_suppression,
                    first_block=(i == 0), layer_type=layer_type
                )
            )
            if i != 0:
                input_size = conv2d_output_shape(input_size, kernel_size=3, stride=2, padding=1)
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def b1(in_channels: int, out_channels: int, layer_type=nn.Conv2d, activation_type=nn.ReLU):
        return nn.Sequential(
            layer_type(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), activation_type(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    @staticmethod
    def block(
            block_index: int, block: ArchBlock, input_size: Tuple[int, int], first_block: bool = False,
            layer_type=nn.Conv2d, activation_type=nn.ReLU, use_suppression: bool = False
    ):
        blk = []
        use_suppression = block_index < 3 and use_suppression
        for i in range(block.num_residuals):
            if i == 0 and not first_block:
                blk.append(
                    Residual(
                        block.prev_num_channels, block.num_channels, input_size, use_suppression=use_suppression,
                        use_1x1conv=True, strides=2, layer_type=layer_type, activation_type=activation_type
                    )
                )
                input_size = conv2d_output_shape(input_size, kernel_size=1, stride=2)
            else:
                blk.append(Residual(
                    block.num_channels, block.num_channels, input_size, use_suppression=use_suppression,
                    layer_type=layer_type, activation_type=activation_type
                ))
        return nn.Sequential(*blk)


class ResNet18(ResNet):
    def __init__(
            self, input_size: Tuple[int, int], num_classes=10, use_suppression: bool = False, layer_type=nn.Conv2d,
            activation_type=nn.ReLU
    ):
        super().__init__(
            input_size,
            [
                ArchBlock(2, 64, 64),
                ArchBlock(2, 64, 128),
                ArchBlock(2, 128, 256),
                ArchBlock(2, 256, 512)
            ],
            num_classes,
            use_suppression=use_suppression,
            layer_type=layer_type,
            activation_type=activation_type,
        )

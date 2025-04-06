"""
Taken from https://d2l.ai/chapter_convolutional-modern/resnet.html
"""
from dataclasses import dataclass
from typing import List

import numpy as np
from torch import nn

from model.layers import SuppressionLayer


class Residual(nn.Module):
    """The Residual block of ResNet models."""

    def __init__(
            self, in_channels: int, out_channels: int, input_size: np.ndarray, use_suppression: bool,
            use_1x1conv: bool = False, strides: int = 1, layer_type=nn.Conv2d, activation_type=nn.ReLU,
    ):
        super().__init__()
        self.input_size = input_size
        suppression_input_size = input_size.copy()
        self.in_channels = in_channels

        self.conv1 = layer_type(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        suppression_input_size = np.ceil(suppression_input_size / strides).astype(int)
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
    input_size: np.ndarray


class ResNet(nn.Module):
    def __init__(
            self, arch: List[ArchBlock], num_classes: int, use_suppression: bool, layer_type=nn.Conv2d,
            activation_type=nn.ReLU
    ):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(self.b1(3, 64, layer_type=layer_type, activation_type=activation_type))
        for i, b in enumerate(arch):
            self.net.add_module(
                f'b{i + 2}',
                self.block(
                    i, b, use_suppression=use_suppression,
                    first_block=(i == 0), layer_type=layer_type
                )
            )
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
            block_index: int, block: ArchBlock, first_block: bool = False,
            layer_type=nn.Conv2d, activation_type=nn.ReLU, use_suppression: bool = False
    ):
        blk = []
        use_suppression = block_index < 3 and use_suppression
        input_size = block.input_size
        for i in range(block.num_residuals):
            if i == 0 and not first_block:
                blk.append(
                    Residual(
                        block.prev_num_channels, block.num_channels, input_size, use_suppression=use_suppression,
                        use_1x1conv=True, strides=2, layer_type=layer_type, activation_type=activation_type
                    )
                )
                input_size = np.ceil(input_size / 2).astype(int)
            else:
                blk.append(Residual(
                    block.num_channels, block.num_channels, input_size, use_suppression=use_suppression,
                    layer_type=layer_type, activation_type=activation_type
                ))
        return nn.Sequential(*blk)


class ResNet18(ResNet):
    def __init__(self, num_classes=10, use_suppression: bool = False, layer_type=nn.Conv2d, activation_type=nn.ReLU):
        super().__init__(
            [
                ArchBlock(2, 64, 64, np.array([24, 24])),
                ArchBlock(2, 64, 128, np.array([24, 24])),
                ArchBlock(2, 128, 256, np.array([12, 12])),
                ArchBlock(2, 256, 512, np.array([6, 6]))
            ],
            num_classes,
            use_suppression=use_suppression,
            layer_type=layer_type,
            activation_type=activation_type,
        )

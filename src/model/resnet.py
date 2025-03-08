"""
Taken from https://d2l.ai/chapter_convolutional-modern/resnet.html
"""

from torch import nn
from torch.nn import functional as func


class Residual(nn.Module):
    """The Residual block of ResNet models."""

    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, x):
        y = func.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return func.relu(y)


class ResNet(nn.Module):
    def __init__(self, arch, num_classes=10):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(self.b1(3, 64))
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i + 2}', self.block(*b, first_block=(i == 0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def b1(in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    @staticmethod
    def block(num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels))
        return nn.Sequential(*blk)


class ResNet18(ResNet):
    def __init__(self, num_classes=10):
        super().__init__(
            [
                (2, 64),
                (2, 128),
                (2, 256),
                (2, 512)
            ],
            num_classes
        )

"""
Taken from https://d2l.ai/chapter_convolutional-modern/resnet.html
"""

from torch import nn

from model.layers import SuppressionLayer


class Residual(nn.Module):
    """The Residual block of ResNet models."""

    def __init__(
            self, in_channels: int, out_channels: int, use_suppression: bool, use_1x1conv: bool = False,
            strides: int = 1, layer_type=nn.Conv2d, activation_type=nn.ReLU,
    ):
        super().__init__()
        self.conv1 = layer_type(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.activation = activation_type()
        self.conv2 = layer_type(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = layer_type(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

        if use_suppression:
            self.suppression = SuppressionLayer(out_channels)
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


class ResNet(nn.Module):
    def __init__(self, arch, num_classes: int, use_suppression: bool, layer_type=nn.Conv2d, activation_type=nn.ReLU):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(self.b1(3, 64, layer_type=layer_type, activation_type=activation_type))
        for i, b in enumerate(arch):
            self.net.add_module(
                f'b{i + 2}',
                self.block(i, *b, use_suppression=use_suppression, first_block=(i == 0), layer_type=layer_type)
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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    @staticmethod
    def block(
            block_index: int, num_residuals, prev_num_channels, num_channels, first_block: bool = False,
            layer_type=nn.Conv2d, activation_type=nn.ReLU, use_suppression: bool = False
    ):
        blk = []
        use_suppression = block_index < 3 and use_suppression
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(
                    Residual(
                        prev_num_channels, num_channels, use_suppression=use_suppression, use_1x1conv=True, strides=2,
                        layer_type=layer_type, activation_type=activation_type
                    )
                )
            else:
                blk.append(Residual(
                    num_channels, num_channels, use_suppression=use_suppression, layer_type=layer_type,
                    activation_type=activation_type
                ))
        return nn.Sequential(*blk)


class ResNet18(ResNet):
    def __init__(self, num_classes=10, use_suppression: bool = False, layer_type=nn.Conv2d, activation_type=nn.ReLU):
        super().__init__(
            [
                (2, 64, 64),
                (2, 64, 128),
                (2, 128, 256),
                (2, 256, 512)
            ],
            num_classes,
            use_suppression=use_suppression,
            layer_type=layer_type,
            activation_type=activation_type,
        )

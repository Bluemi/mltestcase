import torch
import torchsummary

from model.layers import Conv2dMoth
from model.resnet import ResNet18


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(layer_type=Conv2dMoth)
    model.to(device)
    torchsummary.summary(model, (3, 96, 96))

    model_moth = ResNet18(layer_type=Conv2dMoth)
    model_moth.to(device)
    torchsummary.summary(model_moth, (3, 96, 96))


if __name__ == '__main__':
    main()

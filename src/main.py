import torch
import torchsummary

from model.resnet import ResNet18


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ResNet18()
    model.to(device)
    torchsummary.summary(model, (3, 96, 96))


if __name__ == '__main__':
    main()

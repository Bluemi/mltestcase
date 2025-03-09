import os

import torch
import torchsummary
from matplotlib import pyplot as plt
from torchvision import transforms

from model.layers import Conv2dMoth
from model.resnet import ResNet18
from utils import describe
from utils.datasets import ImageNetDataset
from utils.plotting import imshow


SHOW_IMAGES = False


def main():
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = ResNet18(layer_type=Conv2dMoth)
    # model.to(device)
    # torchsummary.summary(model, (3, 96, 96))

    dataset = ImageNetDataset(
        os.path.expanduser('~/data/datasets/ImageNet/'),
        train=True,
        transform=transforms.Resize((224, 224))
    )

    next_label = None
    counter = 0
    for data, label in dataset:
        print(counter, label)
        counter += 1
        if SHOW_IMAGES:
            if next_label is not None:
                if label == next_label:
                    continue
            image = data / 255.0
            key = imshow(image)
            if key == 'escape':
                break
            elif key == 'n':
                next_label = label


if __name__ == '__main__':
    main()

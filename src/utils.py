import os
import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def load_data(batch_size=4, shuffle=True, num_workers=2):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(
        root=os.path.expanduser('~/data/'), train=True, transform=transform
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=os.path.expanduser('~/data/'), train=False, transform=transform
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_data_loader, test_data_loader, classes



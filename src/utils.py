import os
import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm


def imshow(img):
    # img = img / 2 + 0.5
    npimg = img.numpy()
    npimg = np.minimum(np.maximum(npimg, 0.0), 1.0)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def load_data(dataset_name, batch_size=4, shuffle=True, num_workers=2, device=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = {
        'cifar10': torchvision.datasets.CIFAR10,
        'mnist': MnistDataset,
    }[dataset_name]

    # train
    train_dataset = dataset(
        root=os.path.expanduser('~/data/'), download=False, train=True, transform=transform
    )
    if device is not None:
        train_dataset.to_device(device)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False
    )

    # test
    test_dataset = dataset(
        root=os.path.expanduser('~/data/'), download=False, train=False, transform=transform
    )
    if device is not None:
        train_dataset.to_device(device)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False
    )

    classes = {
        'cifar10': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
        'mnist': tuple('0123456789')
    }
    return train_data_loader, test_data_loader, classes


class MnistDataset(Dataset):
    def __init__(self, root, download, train, transform):
        ds = torchvision.datasets.MNIST(
            root=root, download=download, train=train, transform=transform
        )
        self.mnist_data = torch.zeros((len(ds), 1, 28, 28))
        self.mnist_labels = torch.zeros(len(ds))
        for i, sample in tqdm(enumerate(ds), desc="loading mnist", total=len(ds)):
            self.mnist_data[i] = sample[0]
            self.mnist_labels[i] = sample[1]

    def to_device(self, device):
        self.mnist_data = self.mnist_data.to(device)
        self.mnist_labels = self.mnist_labels.to(device)

    def __len__(self):
        return len(self.mnist_labels)

    def __getitem__(self, index):
        return self.mnist_data[index], self.mnist_labels[index]

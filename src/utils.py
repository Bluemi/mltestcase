import os
import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm


def get_mean_std(ds_name):
    return {
        # 'cifar10':
        'mnist': (0.1307, 0.3081)
    }[ds_name]


def get_classes(ds_name):
    return {
        'cifar10': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
        'mnist': tuple('0123456789')
    }[ds_name]


def imshow(img):
    # img = img / 2 + 0.5
    np_img = img.numpy()
    np_img = np.minimum(np.maximum(np_img, 0.0), 1.0)
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def load_data(dataset_name, train, batch_size=4, shuffle=True, num_workers=2, device=None):
    ds_mean, ds_std = get_mean_std(dataset_name)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(ds_mean, ds_std)
    ])

    dataset_c = {
        'cifar10': torchvision.datasets.CIFAR10,
        'mnist': MnistDataset,
    }[dataset_name]

    # train
    dataset = dataset_c(
        root=os.path.expanduser('~/misc/data/'), download=False, train=train, transform=transform
    )
    if device is not None:
        dataset.to_device(device)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False
    )

    return data_loader


class MnistDataset(Dataset):
    def __init__(self, root, download, train, transform):
        ds = torchvision.datasets.MNIST(
            root=root, download=download, train=train, transform=transform
        )
        self.mnist_data = torch.zeros((len(ds), 1, 28, 28))
        self.mnist_labels = torch.zeros(len(ds))
        # noinspection PyTypeChecker
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

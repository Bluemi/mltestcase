import os
from collections import defaultdict
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
try:
    from determined.pytorch import DataLoader
except ImportError:
    from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


POSSIBLE_DATA_DIR_LOCATIONS = [os.path.expanduser('~/misc/data'), '/data', os.path.expanduser('~/data')]


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


def get_image_shapes(ds_name):
    return {
        'mnist': (1, 28, 28),
        'cifar10': (3, 32, 32),
    }[ds_name]


def get_data_dir():
    possible = [p for p in POSSIBLE_DATA_DIR_LOCATIONS if os.path.isdir(p)]
    if not possible:
        raise ValueError('No datadir could be found.')
    return possible[0]


def load_data(dataset_name, train, batch_size=4, shuffle=True, num_workers=2, device=None, use_dataloader=True):
    ds_mean, ds_std = get_mean_std(dataset_name)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(ds_mean, ds_std)
    ])

    dataset_c = {
        'cifar10': torchvision.datasets.CIFAR10,
        'mnist': MnistDataset,
    }[dataset_name]

    datadir = get_data_dir()

    dataset = dataset_c(
        root=datadir, download=True, train=train, transform=transform
    )
    if device is not None:
        dataset.to_device(device)

    if not use_dataloader:
        return dataset

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False
    )


class MnistDataset(Dataset):
    def __init__(self, root, download, train, transform):
        ds = torchvision.datasets.MNIST(
            root=root, download=download, train=train, transform=transform
        )
        self.mnist_data = torch.zeros((len(ds), 1, 28, 28))
        self.mnist_labels = torch.zeros(len(ds), dtype=torch.long)
        # noinspection PyTypeChecker
        for i, sample in tqdm(enumerate(ds), desc="loading mnist", total=len(ds), ascii=True):
            self.mnist_data[i] = sample[0]
            self.mnist_labels[i] = sample[1]

    def to_device(self, device):
        self.mnist_data = self.mnist_data.to(device)
        self.mnist_labels = self.mnist_labels.to(device)

    def __len__(self):
        return len(self.mnist_labels)

    def __getitem__(self, index):
        return self.mnist_data[index], self.mnist_labels[index]


def get_examples(dataset: Dataset, n_labels: int, n: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a tuple [images, labels] of the dataset in which each label is represented equally often.
    The first n images/labels have the first label, the following n images/labels have the second label and so on.

    :param dataset: The dataset to take examples of
    :param n_labels: Number of labels present in the dataset
    :param n: The number of examples for each label
    :return: A tuple containing (images, labels).
    """
    image_shape = next(iter(dataset))[0].shape
    n_examples = n * n_labels

    examples = torch.zeros((n_examples, *image_shape))
    labels = torch.zeros(n_examples, dtype=torch.int)

    counts = defaultdict(int)
    count_sum = 0

    with torch.no_grad():
        for example, label in dataset:
            label = label.item()
            count_for_example = counts[label]
            if count_for_example < n:
                index = label * n + count_for_example

                examples[index] = example
                labels[index] = label

                counts[label] += 1
                count_sum += 1

                if count_sum >= n_examples:
                    break

    if count_sum < n_examples:
        raise ValueError("Could not find enough examples for each label")

    return examples, labels


def get_playground_dataloader(points, labels, batch_size: int, shuffle: bool):
    dataset = PlaygroundDataset(points, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class PlaygroundDataset(Dataset):
    def __init__(self, points, labels):
        self.points = torch.tensor(points, dtype=torch.float32, requires_grad=False)
        self.labels = torch.tensor(labels.reshape(-1, 1), dtype=torch.float32, requires_grad=False)

        assert len(points) == len(labels)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index], self.labels[index]

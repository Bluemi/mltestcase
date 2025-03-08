import torch
import torchvision
from torch.utils.data import Dataset
from tqdm import tqdm


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

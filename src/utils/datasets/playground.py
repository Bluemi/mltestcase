import torch
from torch.utils.data import Dataset, DataLoader


class PlaygroundDataset(Dataset):
    def __init__(self, points, labels):
        self.points = torch.tensor(points, dtype=torch.float32, requires_grad=False)
        self.labels = torch.tensor(labels.reshape(-1, 1), dtype=torch.float32, requires_grad=False)

        assert len(points) == len(labels)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index], self.labels[index]


def get_playground_dataloader(points, labels, batch_size: int, shuffle: bool):
    dataset = PlaygroundDataset(points, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Tuple

import torchvision

import torch


@dataclass
class Label:
    index: int
    identifier: str
    name: str

    def __eq__(self, other):
        return self.index == other.index


@dataclass
class _IndexEntry:
    image_path: str
    label: Label


class ImageNetDataset(torch.utils.data.Dataset):
    MEAN_VALUES = [0.4812, 0.4573, 0.4078]
    STD_VALUES = [0.2161, 0.2117, 0.2126]

    def __init__(self, root: str | Path, transform: Callable, train: bool, full_labels: bool = False):
        """
        :param root: Path to the root directory of the dataset.
        :param transform: A callable object (e.g., torchvision transform) applied to images.
        :param train: Whether to load the training set or the validation set.
        """
        self.root = Path(root)
        self.transform = transform
        self.labels = ImageNetDataset._load_labels(self.root)
        self.image_list = ImageNetDataset._create_image_list(self.root, self.labels, train)
        self.full_labels = full_labels

    @staticmethod
    def _label_name(label: int) -> str:
        pass

    @staticmethod
    def _load_labels(root: Path) -> List[Label]:
        label_path = root / 'ImageSets' / 'CLS-LOC' / 'classes.txt'

        labels = []
        with open(label_path, 'r') as file:
            for index, line in enumerate(file):
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    label = Label(index, parts[0], parts[1].split(", ")[0])
                    labels.append(label)

        return labels

    @staticmethod
    def _create_image_list(root: Path, labels: List[Label], train: bool) -> List[_IndexEntry]:
        index = []
        train_dir = 'train' if train else 'val_sorted'
        images_dir = root / 'Data' / 'CLS-LOC' / train_dir
        for label in labels:
            class_dir = images_dir / label.identifier
            if os.path.isdir(class_dir):
                for image_path in class_dir.iterdir():
                    index.append(_IndexEntry(image_path, label))
            else:
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
        return index
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index) -> Tuple[torch.Tensor, Label]:
        entry = self.image_list[index]

        # image
        image = torchvision.io.read_image(str(entry.image_path), mode=torchvision.io.ImageReadMode.RGB)
        if image.shape[0] == 1:
            image = image.expand(3, *image.shape[1:])
        if image.shape[0] != 3:
            raise ValueError(f"Expected 3 channels, but got {image.shape[0]} for image {entry.image_path}")
        if image.dtype != torch.uint8:
            raise TypeError(f"Expected uint8 dtype, but got {image.dtype} for image {entry.image_path}")
        image = image.to(torch.float32) / 255.0
        image = self.transform(image)

        # label
        label = entry.label
        if not self.full_labels:
            label = label.index

        return image, label

    def get_example_indices(self, n_per_class: int) -> List[int]:
        result_indices = []
        found_per_class = defaultdict(int)
        for index, index_entry in enumerate(self.image_list):
            label_index = index_entry.label.index
            if found_per_class[label_index] < n_per_class:
                found_per_class[label_index] += 1
                result_indices.append(index)
        return result_indices

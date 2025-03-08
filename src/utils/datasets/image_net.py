import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable

from PIL import Image

import torch


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root: str | Path, transform: Callable):
        """
        :param root: Path to the root directory of the dataset.
        :param transform: A callable object (e.g., torchvision transform) applied to images.
        """
        self.root = Path(root)
        self.transform = transform
        self.label_names = ImageNetDataset._load_label_names(root)
        self.index = ImageNetDataset._create_index(root)

    @staticmethod
    def _label_name(label: int) -> str:
        pass

    @staticmethod
    def _load_label_names(root: Path) -> list[str]:
        with open(root / "labels.json", 'r') as f:
            return json.load(f)

    @dataclass
    class _IndexEntry:
        image_path: str
        label: int

    @staticmethod
    def _create_index(root: Path) -> List[_IndexEntry]:
        index = []
        for class_index, class_dir in enumerate(root.iterdir()):
            if os.path.isdir(class_dir):
                for image_path in class_dir.iterdir():
                    index.append(ImageNetDataset._IndexEntry(image_path, class_index))
        return index
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        entry = self.index[index]
        image = Image.open(entry.image_path)
        return self.transform(image), entry.label

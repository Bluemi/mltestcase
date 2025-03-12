import os
from pathlib import Path
from typing import Tuple

from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from utils.datasets import ImageNetDataset


def main():
    dataset = ImageNetDataset(
        os.path.expanduser('~/data/datasets/ImageNet/'),
        train=True,
        transform=transforms.Compose([]),
    )

    new_image_size = 96

    for entry in tqdm(dataset.image_list, desc='scale ImageNet'):
        input_path = Path(entry.image_path)
        output_path = Path(str(entry.image_path).replace("/ImageNet/", f"/ImageNet_{new_image_size}/", 1))
        if not output_path.exists():
            resize_and_save_image(input_path, output_path, (new_image_size, new_image_size))


def resize_and_save_image(input_path: Path, output_path: Path, size: Tuple[int, int]):
    """
    Loads an image from input_path, scales it to the given size, and writes it to output_path.
    Ensures all necessary directories are created.

    :param input_path: Path to the input image.
    :param output_path: Path to save the resized image.
    :param size: Tuple specifying the new size (width, height).
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(input_path)
    image = image.resize(size, Image.Resampling.LANCZOS)
    image.save(output_path)


if __name__ == '__main__':
    main()

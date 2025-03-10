import os

import torch
import torchsummary
from torchvision import transforms
from tqdm import tqdm

from model.layers import Conv2dMoth
from model.resnet import ResNet18
from utils import describe
from utils.datasets import ImageNetDataset
from utils.plotting import imshow


SHOW_IMAGES = True
NUM_SAMPLES_PER_CLASS = 100


def show_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(layer_type=Conv2dMoth)
    model.to(device)
    torchsummary.summary(model, (3, 96, 96))


def show_dataset():
    dataset = ImageNetDataset(
        os.path.expanduser('~/data/datasets/ImageNet/'),
        train=True,
        transform=transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.Normalize([122.7, 116.6, 104.0], [55.1, 54.0, 54.2])
        ]),
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


def calculate_mean_std():
    dataset = ImageNetDataset(
        os.path.expanduser('~/data/datasets/ImageNet/'),
        train=False,
        transform=transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.Normalize(ImageNetDataset.MEAN_VALUES, ImageNetDataset.STD_VALUES)
        ]),
    )

    mean_sum = torch.zeros(3, dtype=torch.float64)
    std_sum = torch.zeros(3, dtype=torch.float64)

    counter = 0

    for index in dataset.get_example_indices(NUM_SAMPLES_PER_CLASS):
        data, label = dataset[index]
        data = data.to(torch.float64)
        print(label)
        describe(data, 'data')

        mean_sum += torch.mean(data, dim=(1, 2))
        std_sum += torch.std(data, dim=(1, 2))

        counter += 1

    print('mean: ', mean_sum / counter)
    print('std: ', std_sum / counter)


def clear_dataset():
    dataset = ImageNetDataset(
        os.path.expanduser('~/data/datasets/ImageNet/'),
        train=True,
        transform=transforms.Compose([]),
    )

    for data, label in tqdm(dataset):
        if data.shape[0] != 3:
            print(dataset.image_list[label.index].image_path, data.shape)


if __name__ == '__main__':
    calculate_mean_std()
    # clear_dataset()

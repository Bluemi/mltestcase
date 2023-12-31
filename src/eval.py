import torch
import torchvision

from model import MnistAutoencoder
from train import MODEL_PATH
from utils.datasets import load_data, get_mean_std
from utils import imshow, denormalize


def main():
    net = MnistAutoencoder()
    net.load_state_dict(torch.load(MODEL_PATH))

    dataset = load_data('mnist', train=False, batch_size=8, num_workers=0)

    with torch.no_grad():
        for data, labels in dataset:
            outputs = net(data)
            outputs = torch.reshape(outputs, (-1, 1, 28, 28))

            inputs = data.cpu()

            show_image = torch.concat([inputs, outputs])
            ds_mean, ds_std = get_mean_std('mnist')
            show_image = denormalize(show_image, ds_mean, ds_std)
            if imshow(torchvision.utils.make_grid(show_image)) == 'escape':
                break


if __name__ == '__main__':
    main()

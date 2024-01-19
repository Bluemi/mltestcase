import argparse

import torch
import torchvision

from model import MnistAutoencoder
from utils.datasets import load_data, get_mean_std
from utils import imshow, denormalize
from utils.evaluation import model_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate model on mnist')
    parser.add_argument('model_path', type=str, help='Path the model to evaluate.')

    return parser.parse_args()


def main():
    args = parse_args()
    model = MnistAutoencoder()
    model.load_state_dict(torch.load(args.model_path))

    dataset = load_data('mnist', train=False, batch_size=8, num_workers=0)

    accuracy = model_accuracy(model, dataset)
    print('accuracy: {}'.format(accuracy))

    show_prediction_images(dataset, model)


def show_prediction_images(dataset, net):
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

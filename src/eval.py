import argparse

import torch
import torchvision

from model import MnistAutoencoder
from utils.datasets import load_data, get_mean_std
from utils import imshow, denormalize, fourier_transform_2d, inv_fourier_transform_2d
from utils.evaluation import model_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate model on mnist')
    parser.add_argument('model_path', type=str, help='Path the model to evaluate.')
    parser.add_argument('--fft', action='store_true', help='If set, model is trained on fft output.')

    return parser.parse_args()


def main():
    args = parse_args()
    model = MnistAutoencoder()
    model.load_state_dict(torch.load(args.model_path))

    dataset = load_data('mnist', train=False, batch_size=8, num_workers=0)

    accuracy = model_accuracy(model, dataset, use_fft=args.fft)
    print('accuracy: {}'.format(accuracy))

    show_prediction_images(dataset, model, use_fft=args.fft)


def show_prediction_images(dataset, model, use_fft=False):
    with torch.no_grad():
        for data, labels in dataset:
            inputs = data.cpu()
            if use_fft:
                data = fourier_transform_2d(data)
            outputs = model(data)
            outputs = torch.reshape(outputs, (-1, 1, 28, 28))

            if use_fft:
                outputs = inv_fourier_transform_2d(outputs)

            show_image = torch.concat([inputs, outputs])
            ds_mean, ds_std = get_mean_std('mnist')
            show_image = denormalize(show_image, ds_mean, ds_std)
            if imshow(torchvision.utils.make_grid(show_image)) == 'escape':
                break


if __name__ == '__main__':
    main()

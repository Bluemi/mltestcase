import argparse

import torch
import torchvision

from model import MnistAutoencoder
from utils.datasets import load_data, get_mean_std
from utils import imshow, denormalize, fourier_transform_2d, inv_fourier_transform_2d, cosine_transform_2d, \
    inv_cosine_transform_2d
from utils.evaluation import model_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate model on mnist')
    parser.add_argument('model_path', type=str, help='Path the model to evaluate.')
    parser.add_argument(
        '--ft', default=None, choices=['fft', 'dct'],
        help='Either "fft" or "dct". If set, model is trained on fft/dct output.'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    model = MnistAutoencoder()
    model.load_state_dict(torch.load(args.model_path))

    dataset = load_data('mnist', train=False, batch_size=8, num_workers=0)

    accuracy = model_accuracy(model, dataset, use_ft=args.ft)
    print('accuracy: {}'.format(accuracy))

    show_prediction_images(dataset, model, use_ft=args.ft)


def show_prediction_images(dataset, model, use_ft=False):
    with torch.no_grad():
        for data, labels in dataset:
            inputs = data.cpu()
            if use_ft == 'fft':
                data = fourier_transform_2d(data)
            elif use_ft == 'dct':
                data = cosine_transform_2d(data)
            outputs = model(data)
            outputs = torch.reshape(outputs, (-1, 1, 28, 28))

            if use_ft == 'fft':
                outputs = inv_fourier_transform_2d(outputs)
            elif use_ft == 'dct':
                outputs = inv_cosine_transform_2d(outputs)

            show_image = torch.concat([inputs, outputs])
            ds_mean, ds_std = get_mean_std('mnist')
            show_image = denormalize(show_image, ds_mean, ds_std)
            if imshow(torchvision.utils.make_grid(show_image)) == 'escape':
                break


if __name__ == '__main__':
    main()

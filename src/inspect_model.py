import argparse

import torch
import torchvision.utils
from torch import nn

from model import MnistAutoencoder, BlobLayer
from utils.plotting import plot_3d_tensor, imshow


def parse_args():
    parser = argparse.ArgumentParser(description='inspect model on mnist')
    parser.add_argument('model_path', type=str, help='Path the model to evaluate.')
    parser.add_argument('--blob-layer', action='store_true', help='Use blob layer as first layer. Otherwise use Linear layer.')

    return parser.parse_args()


def inspect_blob_layer(blob_layer):
    curves = blob_layer.calc_curves()
    curves = torch.moveaxis(curves, 2, 0).detach().numpy()
    for curve in curves:
        if plot_3d_tensor(curve) == 'escape':
            break


def inspect_linear_layer(linear_layer):
    weight_images = linear_layer.weight
    if weight_images.shape[1] == 28 * 28:
        visualize_weights_matrix(weight_images)
    elif weight_images.shape[0] == 28 * 28:
        visualize_weights_matrix(weight_images.T)


def visualize_weights_matrix(weight_images):
    images = []
    aborted = False
    for weight_image in weight_images:
        image = weight_image.detach()
        image = image.reshape(1, 28, 28)
        images.append(image)
        if len(images) >= 16:
            if imshow(torchvision.utils.make_grid(images)) == 'escape':
                aborted = True
                break
            images = []

    if len(images) > 0 and not aborted:
        imshow(torchvision.utils.make_grid(images))


def main():
    args = parse_args()

    # load model
    device = torch.device('cpu')
    model = MnistAutoencoder(use_blob_layer=args.blob_layer)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    first_layer = model.encoder[0]
    if isinstance(first_layer, nn.Linear):
        inspect_linear_layer(first_layer)
    elif isinstance(first_layer, BlobLayer):
        inspect_blob_layer(first_layer)

    last_layer = model.decoder[-1]
    inspect_linear_layer(last_layer)


if __name__ == '__main__':
    main()

import argparse

import torch

from model import MnistAutoencoder
from utils.plotting import plot_3d_tensor


def parse_args():
    parser = argparse.ArgumentParser(description='inspect model on mnist')
    parser.add_argument('model_path', type=str, help='Path the model to evaluate.')
    parser.add_argument('--blob-layer', action='store_true', help='Use blob layer as first layer. Otherwise use Linear layer.')

    return parser.parse_args()


def main():
    args = parse_args()

    # load model
    device = torch.device('cpu')
    model = MnistAutoencoder(use_blob_layer=args.blob_layer)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    layer = model.encoder[0]
    curves = layer.calc_curves()
    curves = torch.moveaxis(curves, 2, 0).detach().numpy()
    for curve in curves:
        if plot_3d_tensor(curve) == 'escape':
            break


if __name__ == '__main__':
    main()

import sys

import torch

from train import MODEL_PATH
from utils.datasets import load_data, get_examples, get_classes, get_mean_std
from utils.interactive_visualizations import Vec2Img
from model import MnistAutoencoder


def main():
    model_path = MODEL_PATH
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    model = MnistAutoencoder()
    # model.load_state_dict(torch.load('models/mnist_autoencoder_custom_loss.pth'), strict=False)
    model.load_state_dict(torch.load(model_path), strict=False)

    dataset = load_data('mnist', train=False, batch_size=8, num_workers=0, use_dataloader=False)
    samples = get_examples(dataset, len(get_classes('mnist')), n=500)

    normalization_mean_std = get_mean_std('mnist')
    window = Vec2Img(model, samples, normalization_mean_std=normalization_mean_std)
    window.run()


if __name__ == '__main__':
    main()

import torch

from train import MODEL_PATH
from utils.datasets import load_data, get_examples, get_classes
from utils.interactive_visualizations import Vec2Img
from model import MnistAutoencoder


def main():
    model = MnistAutoencoder()
    model.load_state_dict(torch.load(MODEL_PATH))

    dataset = load_data('mnist', train=False, batch_size=8, num_workers=0, use_dataloader=False)
    samples = get_examples(dataset, len(get_classes('mnist')), n=10)

    window = Vec2Img(model, samples)
    window.run()


if __name__ == '__main__':
    main()

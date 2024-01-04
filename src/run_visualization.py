import torch

from train import MODEL_PATH
from utils.interactive_visualizations import Vec2Img
from model import MnistAutoencoder


def main():
    model = MnistAutoencoder()
    model.load_state_dict(torch.load(MODEL_PATH))

    window = Vec2Img(model)
    window.run()


if __name__ == '__main__':
    main()

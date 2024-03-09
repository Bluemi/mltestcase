from torch.nn import Module, Sequential, Linear, Sigmoid
from model.layers import MothLayer


class PlaygroundModel(Module):
    def __init__(self, training=False):
        super().__init__()
        self.training = training

        self.layers = Sequential(
            Linear(2, 8),
            MothLayer(8),
            # Sigmoid(),
            Linear(8, 4),
            MothLayer(4),
            # Sigmoid(),
            Linear(4, 1),
        )

    def forward(self, x):
        return self.layers.forward(x)
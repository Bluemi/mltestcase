from torch.nn import Module, Sequential, Linear, Sigmoid


class PlaygroundModel(Module):
    def __init__(self, training=False):
        super().__init__()
        self.training = training

        self.layers = Sequential(
            Linear(2, 4),
            Sigmoid(),
            Linear(4, 2),
            Sigmoid(),
            Linear(2, 1),
        )

    def forward(self, x):
        return self.layers.forward(x)
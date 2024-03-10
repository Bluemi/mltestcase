from torch.nn import Module, Sequential, Linear, Sigmoid
from model.layers import MothLayer


class PlaygroundModel(Module):
    def __init__(self, training=False, activation_function='sigmoid'):
        super().__init__()
        self.training = training

        if activation_function == 'sigmoid':
            activation_func = lambda n: Sigmoid()
        elif activation_function == 'moth':
            activation_func = MothLayer
        else:
            raise ValueError('Unknown activation function: {}'.format(activation_function))

        self.layers = Sequential(
            Linear(2, 8),
            activation_func(8),
            Linear(8, 4),
            activation_func(4),
            Linear(4, 2),
            activation_func(2),
            Linear(2, 1),
        )

    def forward(self, x):
        return self.layers.forward(x)
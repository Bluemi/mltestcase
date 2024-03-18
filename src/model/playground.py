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

        layer_sizes = [
            2, 16, 8, 1
        ]

        layers = []

        for index, sizes in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            input_size, output_size = sizes
            layers.append(Linear(input_size, output_size))
            if index != len(layer_sizes) - 2:
                layers.append(activation_func(output_size))

        self.layers = Sequential(*layers)

    def forward(self, x):
        return self.layers.forward(x)
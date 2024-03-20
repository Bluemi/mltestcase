import argparse
import time
from typing import Optional

import torch
import torch.optim as optim
from torch import nn
from torchsummary import summary
from tqdm import trange

from model.mnist import MnistAutoencoder
from utils import fourier_transform_2d, cosine_transform_2d
from utils.datasets import load_data
from utils.loss_functions import custom_loss_function

BATCH_SIZE = 512
NUM_EPOCHS = 150
EMBEDDING_SIZE = 2
MODEL_PATH = 'models/mnist_classifier.pth'
LEARNING_RATE = 0.007


def parse_args():
    parser = argparse.ArgumentParser(description='train model on mnist')
    parser.add_argument(
        'save_path', type=str, default=None, nargs='?',
        help='The path where the trained model will be saved. If not specified will not save any model.'
    )
    parser.add_argument('--init', type=str, default=None, help='The model to load as starting point')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='The learning rate used for training.')
    parser.add_argument('--wc', type=float, default=0.0015, help='The weight decay used for training.')
    parser.add_argument('--momentum', '-m', type=float, default=0.0, help='The momentum of the sgd optimizer used for training.')
    parser.add_argument(
        '--ft', default=None, choices=['fft', 'dct'],
        help='Either "fft" or "dct". If set, model is trained on fft/dct output.'
    )
    parser.add_argument('--blob-layer', action='store_true', help='Use blob layer as first layer. Otherwise use Linear layer.')
    parser.add_argument('--moth-layer', action='store_true', help='Use moth layer as activation function. Otherwise use Sigmoid.')
    parser.add_argument('--epochs', '-e', type=int, default=NUM_EPOCHS, help=f'The number of epochs. Defaults to {NUM_EPOCHS}.')
    parser.add_argument('--autoencoder', action='store_true', help='Train with autoencoder loss')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()

    activation_func = 'sigmoid'
    if args.moth_layer:
        activation_func = 'moth'
    model = MnistAutoencoder(use_blob_layer=args.blob_layer, activation_func=activation_func)
    print('loading model: \"{}\"'.format(args.init))
    if args.init:
        model.load_state_dict(torch.load(args.init), strict=False)
    model.to(device)

    print('save model to: \"{}\"'.format(args.save_path))

    summary(model, input_size=(1, 28, 28))

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wc)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wc, momentum=args.momentum)

    train_dataset = load_data('mnist', train=True, batch_size=BATCH_SIZE, num_workers=0, device=device)

    last_loss = train(
        train_dataset, model, optimizer, device, save_path=args.save_path, use_ft=args.ft, epochs=args.epochs,
        train_autoencoder=args.autoencoder
    )

    print('lr={} gives loss={}'.format(LEARNING_RATE, last_loss))
    print(f'training took {time.time() - start_time} seconds.')


def train(
        train_dataset, model, optimizer, device, save_path: Optional[str] = None, use_ft=None, epochs=NUM_EPOCHS,
        train_autoencoder=False
):
    last_loss = None
    pbar = trange(epochs, ascii=True, desc=f'l={0.0:.4f}')
    for _epoch in pbar:
        current_loss_sum = 0.0
        example_counter = 0
        for data in train_dataset:
            inputs, labels = data
            optimizer.zero_grad()

            if use_ft == 'fft':
                with torch.no_grad():
                    inputs = fourier_transform_2d(inputs)
            elif use_ft == 'dct':
                with torch.no_grad():
                    inputs = cosine_transform_2d(inputs.cpu()).to(device)

            if train_autoencoder:
                embedding = model.encode(inputs)
                outputs = model.decode(embedding)
                predictions = model.classification_head(embedding)

                autoencoder_loss = calc_autoencoder_loss(inputs, outputs, embedding, labels)
                autoencoder_coefficient = _get_autoencoder_coefficient(use_ft)
                classifier_loss = calc_classifier_loss(predictions, labels)
                loss = autoencoder_coefficient * autoencoder_loss + classifier_loss
            else:
                predictions = model(inputs)
                loss = calc_classifier_loss(predictions, labels)


            loss.backward()
            optimizer.step()

            current_loss_sum += loss.item()
            example_counter += 1
        last_loss = current_loss_sum / example_counter
        pbar.set_description(f'l={last_loss:.4f}')

    if save_path:
        torch.save(model.state_dict(), save_path)

    return last_loss


def calc_classifier_loss(predictions, labels):
    return nn.functional.cross_entropy(predictions, labels)


def calc_autoencoder_loss(inputs, outputs, embedding, labels):
    return custom_loss_function(
        outputs, torch.flatten(inputs, start_dim=1), embedding, labels, alpha=0.05, beta=1.5, gamma=3.0
    )


def _get_autoencoder_coefficient(use_ft):
    autoencoder_coefficient = 0.2
    if use_ft == 'fft':
        autoencoder_coefficient = 0.004
    elif use_ft == 'dct':
        autoencoder_coefficient = 0.02
    return autoencoder_coefficient




if __name__ == '__main__':
    main()

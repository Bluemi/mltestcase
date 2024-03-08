import argparse
import time
from typing import Optional

import torch
import torch.optim as optim
from torch import nn
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


def calc_classifier_loss(model, inputs, labels):
    predictions = model.forward_classify(inputs)
    return nn.functional.cross_entropy(predictions, labels)


def calc_autoencoder_loss(model, inputs, labels):
    embedding = model.encode(inputs)
    outputs = model.decode(embedding)

    return custom_loss_function(
        outputs, torch.flatten(inputs, start_dim=1), embedding, labels, beta=1.0, gamma=2.0
    )


def train(train_dataset, model, optimizer, device, save_path: Optional[str] = None, use_ft=None):
    last_loss = None
    pbar = trange(NUM_EPOCHS, ascii=True, desc=f'l={0.0:.4f}')
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

            autoencoder_loss = calc_autoencoder_loss(model, inputs, labels)
            classifier_loss = calc_classifier_loss(model, inputs, labels)
            autoencoder_coefficient = 0.2
            if use_ft == 'fft':
                autoencoder_coefficient = 0.004
            elif use_ft == 'dct':
                autoencoder_coefficient = 0.02
            loss = autoencoder_coefficient * autoencoder_loss + classifier_loss

            loss.backward()
            optimizer.step()

            current_loss_sum += loss.item()
            example_counter += 1
        last_loss = current_loss_sum / example_counter
        pbar.set_description(f'l={last_loss:.4f}')

    if save_path:
        torch.save(model.state_dict(), save_path)

    return last_loss


def parse_args():
    parser = argparse.ArgumentParser(description='train model on mnist')
    parser.add_argument(
        'save_path', type=str, default=None, nargs='?',
        help='The path where the trained model will be saved. If not specified will not save any model.'
    )
    parser.add_argument('--init', type=str, default=None, help='The model to load as starting point')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='The learning rate used for training.')
    parser.add_argument('--wc', type=float, default=0.0015, help='The weight decay used for training.')
    parser.add_argument(
        '--ft', default=None, choices=['fft', 'dct'],
        help='Either "fft" or "dct". If set, model is trained on fft/dct output.'
    )
    parser.add_argument('--blob-layer', action='store_true', help='Use blob layer as first layer. Otherwise use Linear layer.')
    parser.add_argument('--epochs', '-e', type=int, default=NUM_EPOCHS, help=f'The number of epochs. Defaults to {NUM_EPOCHS}.')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()

    model = MnistAutoencoder(use_blob_layer=args.blob_layer)
    print('loading model: \"{}\"'.format(args.init))
    if args.init:
        model.load_state_dict(torch.load(args.init), strict=False)
    model.to(device)

    print('save model to: \"{}\"'.format(args.save_path))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wc)

    train_dataset = load_data('mnist', train=True, batch_size=BATCH_SIZE, num_workers=0, device=device)

    last_loss = train(train_dataset, model, optimizer, device, save_path=args.save_path, use_ft=args.ft)

    print('lr={} gives loss={}'.format(LEARNING_RATE, last_loss))
    print(f'training took {time.time() - start_time} seconds.')


if __name__ == '__main__':
    main()

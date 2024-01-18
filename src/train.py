import argparse
import time
from typing import Optional

import torchvision
from tqdm import trange

from model import MnistAutoencoder
from utils import imshow
from utils.datasets import load_data
import torch
import torch.optim as optim
from torch import nn


BATCH_SIZE = 512
NUM_EPOCHS = 150
EMBEDDING_SIZE = 2
MODEL_PATH = 'models/mnist_classifier.pth'
LEARNING_RATE = 0.007


def same_loss(diffs, labels: torch.Tensor):
    same_mask = torch.eq(labels.reshape(-1, 1), labels.reshape(1, -1))
    masked_diff = same_mask * diffs
    s_loss = torch.sum(masked_diff) / torch.sum(same_mask)
    return s_loss


def different_loss(diffs, labels, sigma):
    different_mask = torch.ne(labels.reshape(1, -1), labels.reshape(-1, 1))
    masked_diff = torch.exp(-different_mask.to(int) * diffs * sigma)
    d_loss = torch.sum(masked_diff) / torch.sum(different_mask)
    return d_loss


def custom_loss_function(outputs, inputs, embedding, labels, beta=1.0, gamma=1.0, sigma=0.5):
    batch_size = outputs.size(0)
    diffs = embedding.reshape(batch_size, 1, EMBEDDING_SIZE) - embedding.reshape(1, batch_size, EMBEDDING_SIZE)
    diffs = torch.sum(torch.square(diffs), dim=2)
    s_loss = same_loss(diffs, labels)
    d_loss = different_loss(diffs, labels, sigma)
    mse_loss = torch.mean(torch.square(outputs - inputs))
    loss = mse_loss + beta * s_loss + gamma * d_loss
    # print(f'loss={loss} mse_loss={mse_loss} s_loss={s_loss * beta}, d_loss={d_loss * gamma}')
    return loss


def calc_classifier_loss(model, inputs, labels):
    predictions = model.forward_classify(inputs)
    return nn.functional.cross_entropy(predictions, labels)


def calc_autoencoder_loss(model, inputs, labels):
    embedding = model.encode(inputs)
    outputs = model.decode(embedding)

    return custom_loss_function(
        outputs, torch.flatten(inputs, start_dim=1), embedding, labels, beta=1.0, gamma=2.0
    )


def train(train_dataset, net, optimizer, save_path: Optional[str] = None):
    last_loss = None
    pbar = trange(NUM_EPOCHS, ascii=True, desc=f'l={0.0:.4f}')
    for _epoch in pbar:
        current_loss_sum = 0.0
        example_counter = 0
        for data in train_dataset:
            inputs, labels = data
            optimizer.zero_grad()

            autoencoder_loss = calc_autoencoder_loss(net, inputs, labels)
            classifier_loss = calc_classifier_loss(net, inputs, labels)
            # print(f'autoencoder_loss: {autoencoder_loss:.4f}, classifier_loss: {classifier_loss:.4f}')
            loss = 0.2 * autoencoder_loss + classifier_loss

            loss.backward()
            optimizer.step()

            current_loss_sum += loss.item()
            example_counter += 1
        last_loss = current_loss_sum / example_counter
        pbar.set_description(f'l={last_loss:.4f}')

    if save_path:
        torch.save(net.state_dict(), save_path)

    return last_loss


def test_model(test_dataset, net, device):
    with torch.no_grad():
        for data in test_dataset:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            outputs = torch.reshape(outputs, (-1, 1, 28, 28)).cpu()

            imshow(torchvision.utils.make_grid(outputs))

            break


def parse_args():
    parser = argparse.ArgumentParser(description='train model on mnist')
    parser.add_argument(
        'save_path', type=str, default=None, nargs='?',
        help='The path where the trained model will be saved. If not specified will not save any model.'
    )
    parser.add_argument('--init', type=str, default=None, help='The model to load as starting point')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='The learning rate used for training.')
    parser.add_argument('--wc', type=float, default=0.0015, help='The weight decay used for training.')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    model = MnistAutoencoder()

    print('loading model: \"{}\"'.format(args.init))
    if args.init:
        model.load_state_dict(torch.load(args.init), strict=False)
    model.to(device)

    print('save model to: \"{}\"'.format(args.save_path))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wc)

    train_dataset = load_data('mnist', train=True, batch_size=BATCH_SIZE, num_workers=0, device=device)

    last_loss = train(train_dataset, model, optimizer, save_path=args.save_path)
    print('lr={} gives loss={}'.format(LEARNING_RATE, last_loss))
    print(f'training took {time.time() - start_time} seconds.')


if __name__ == '__main__':
    main()

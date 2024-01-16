import sys
import time

import torchvision

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
    diffs = torch.sum(torch.square(diffs), axis=2)
    s_loss = same_loss(diffs, labels)
    d_loss = different_loss(diffs, labels, sigma)
    mse_loss = torch.mean(torch.square(outputs - inputs))
    loss = mse_loss + beta * s_loss + gamma * d_loss
    print(f'loss={loss} mse_loss={mse_loss} s_loss={s_loss * beta}, d_loss={d_loss * gamma}')
    return loss


def train(train_dataset, net, optimizer, save_model=True):
    last_loss = None
    # for _epoch in trange(NUM_EPOCHS, ascii=True, desc='train with lr={:.2f}'.format(lr)):
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(NUM_EPOCHS):
        current_loss_sum = 0.0
        example_counter = 0
        for data in train_dataset:
            inputs, labels = data
            optimizer.zero_grad()

            # embedding = net.encode(inputs)
            # outputs = net.decode(embedding)

            # loss = custom_loss_function(
            #     outputs, torch.flatten(inputs, start_dim=1), embedding, labels, beta=1.0, gamma=2.0
            # )
            predictions = net.forward_classify(inputs)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()

            current_loss_sum += loss.item()
            example_counter += 1
        last_loss = current_loss_sum / example_counter
        print(f'Epoch {epoch+1}: {last_loss}')

    if save_model:
        torch.save(net.state_dict(), MODEL_PATH)

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


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset = load_data('mnist', train=True, batch_size=BATCH_SIZE, num_workers=0, device=device)

    start_time = time.time()
    net = MnistAutoencoder()
    net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=0.0015)

    last_loss = train(train_dataset, net, optimizer, save_model=True)
    print('lr={} gives loss={}'.format(LEARNING_RATE, last_loss))
    print(f'training took {time.time() - start_time} seconds.')


if __name__ == '__main__':
    main()

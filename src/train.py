import time

import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

from model import MnistAutoencoder
from utils import imshow
from utils.datasets import load_data
import torch
import torch.optim as optim
from torch import nn


BATCH_SIZE = 512
NUM_EPOCHS = 200
MODEL_PATH = 'models/mnist_autoencoder.pth'
# LEARNING_RATES = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
LEARNING_RATES = [0.02]


def train(train_dataset, net, optimizer, loss_function, lr_scheduler, save_model=True):
    last_loss = None
    for _epoch in trange(NUM_EPOCHS, ascii=True, desc='train with lr={:.2f}'.format(lr_scheduler.get_last_lr()[0])):
        current_loss_sum = 0.0
        example_counter = 0
        for data in train_dataset:
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = loss_function(outputs, torch.flatten(inputs, start_dim=1))
            loss.backward()
            optimizer.step()

            current_loss_sum += loss.item()
            example_counter += 1
        # print(f'Loss {epoch+1}: {current_loss_sum / example_counter}  LR: {lr_scheduler.get_last_lr()}')
        last_loss = current_loss_sum / example_counter
        lr_scheduler.step()

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
    for lr in LEARNING_RATES:
        net = MnistAutoencoder()
        net.to(device)

        loss_function = nn.MSELoss()
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
        lr_scheduler = CosineAnnealingLR(optimizer, NUM_EPOCHS, 0.0002)
        last_loss = train(train_dataset, net, optimizer, loss_function, lr_scheduler, save_model=True)
        print('lr={} gives loss={}'.format(lr, last_loss))
    print(f'training took {time.time() - start_time} seconds.')

    # test_model(train_dataset, net, device)


if __name__ == '__main__':
    main()

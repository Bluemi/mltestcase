import time

import torchvision
from tqdm import tqdm

from model import Net, DenseNetMnist
from utils import load_data, imshow
import torch
import torch.optim as optim
from torch import nn


BATCH_SIZE = 32
NUM_EPOCHS = 50
MODEL_PATH = 'models/cifar_net.pth'


def train(train_dataset, net, optimizer, loss_function, device, save_model=True):
    for epoch in range(NUM_EPOCHS):
        current_loss_sum = 0.0
        example_counter = 0
        for data in tqdm(train_dataset, desc=f'Epoch {epoch+1}', ascii=True):
            # inputs, labels = data[0].to(device), data[1].to(device)
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = loss_function(outputs, torch.flatten(inputs, start_dim=1))
            loss.backward()
            optimizer.step()

            current_loss_sum += loss.item()
            example_counter += 1
        print(f'Loss {epoch+1}: {current_loss_sum / example_counter}')

    if save_model:
        torch.save(net.state_dict(), MODEL_PATH)


def test_model(test_dataset, net, classes, device):
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
    train_dataset, test_dataset, classes = load_data('mnist', batch_size=BATCH_SIZE, num_workers=0, device=device)
    net = DenseNetMnist()
    net.to(device)

    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    start_time = time.time()
    train(train_dataset, net, optimizer, loss_function, save_model=True, device=device)
    print(f'training took {time.time() - start_time} seconds.')

    test_model(test_dataset, net, classes, device)


if __name__ == '__main__':
    main()

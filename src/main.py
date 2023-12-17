import time

import numpy as np
from tqdm import tqdm

from model import Net, DenseNetMnist
from utils import load_data
import torch
import torch.optim as optim
from torch import nn


BATCH_SIZE = 32
NUM_EPOCHS = 10
MODEL_PATH = 'models/cifar_net.pth'


def train(train_dataset, net, optimizer, loss_function, device, save_model=True):
    for epoch in range(NUM_EPOCHS):
        current_loss_sum = 0.0
        example_counter = 0
        for data in tqdm(train_dataset, desc=f'Epoch {epoch+1}', ascii=True):
            inputs, labels = data[0].to(device), data[1].to(device)
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
    correct = 0
    total = 0

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataset:
            labels: np.ndarray  # to prevent ide warning
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset, test_dataset, classes = load_data('mnist', batch_size=BATCH_SIZE, num_workers=2)
    net = DenseNetMnist()
    net.to(device)

    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    start_time = time.time()
    train(train_dataset, net, optimizer, loss_function, save_model=True, device=device)
    print(f'training took {time.time() - start_time} seconds.')

    test_model(test_dataset, net, classes, device)


if __name__ == '__main__':
    main()

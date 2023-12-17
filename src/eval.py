import torch
import torchvision

from model import DenseNetMnist
from train import MODEL_PATH, BATCH_SIZE
from utils import load_data, imshow


def main():
    net = DenseNetMnist()
    net.load_state_dict(torch.load(MODEL_PATH))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset, test_dataset, classes = load_data('mnist', batch_size=8, num_workers=0, device=device)

    with torch.no_grad():
        for data, labels in test_dataset:
            outputs = net(data)
            outputs = torch.reshape(outputs, (-1, 1, 28, 28)).cpu()

            inputs = data.cpu()
            show_image = torch.concat([inputs, outputs])
            imshow(torchvision.utils.make_grid(show_image))


if __name__ == '__main__':
    main()

import torch
import torchvision

from model import DenseNetMnist
from train import MODEL_PATH
from utils import load_data, imshow, get_mean_std


def main():
    net = DenseNetMnist()
    net.load_state_dict(torch.load(MODEL_PATH))

    dataset = load_data('mnist', train=False, batch_size=8, num_workers=0)

    with torch.no_grad():
        for data, labels in dataset:
            outputs = net(data)
            outputs = torch.reshape(outputs, (-1, 1, 28, 28))

            inputs = data.cpu()

            show_image = torch.concat([inputs, outputs])
            ds_mean, ds_std = get_mean_std('mnist')
            show_image = show_image * ds_std + ds_mean
            imshow(torchvision.utils.make_grid(show_image))


if __name__ == '__main__':
    main()

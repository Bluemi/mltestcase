import torch

from utils import fourier_transform_2d, describe


def main():
    img = torch.rand(1, 28, 28)

    describe(img, 'img')

    result = fourier_transform_2d(img)

    describe(result, 'result')


if __name__ == '__main__':
    main()

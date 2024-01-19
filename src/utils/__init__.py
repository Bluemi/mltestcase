import numpy as np
import torch
from matplotlib import pyplot as plt


pressed_key = None


def _press(event):
    global pressed_key
    pressed_key = event.key


def imshow(img):
    # img = img / 2 + 0.5
    np_img = img.numpy()
    np_img = np.minimum(np.maximum(np_img, 0.0), 1.0)

    plt.ion()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))

    plt.show()
    # noinspection PyTypeChecker
    plt.gcf().canvas.mpl_connect('key_press_event', _press)
    plt.waitforbuttonpress()
    plt.close()

    return pressed_key


def describe(x, label):
    if isinstance(x, torch.Tensor):
        print(label)
        print(f'  shape={list(x.shape)}  dtype={x.dtype}')
        if x.dtype != torch.cfloat:
            minimum = x.min().item()
            maximum = x.max().item()
        else:
            minimum = x.real.min().item() + x.imag.min().item()*1j
            maximum = x.real.max().item() + x.imag.max().item()*1j
        print(f'  mean={x.mean():.4f}  min={minimum:.4f}  max={maximum:.4f}')
    else:
        print(f'no description for type \"{type(x).__name__}\"')


def denormalize(data, mean, std):
    return data * std + mean


def fourier_transform_2d(img):
    return torch.fft.fft2(img)

import numpy as np
import scipy
import torch


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
    elif isinstance(x, np.ndarray):
        print(label)
        print(f'  shape={list(x.shape)}  dtype={x.dtype}')
        if not np.iscomplexobj(x):
            minimum = x.min()
            maximum = x.max()
        else:
            minimum = x.real.min() + x.imag.min() * 1j
            maximum = x.real.max() + x.imag.max() * 1j
        print(f'  mean={x.mean():.4f}  min={minimum:.4f}  max={maximum:.4f}')
    else:
        print(f'no description for type \"{type(x).__name__}\"')


def denormalize(data, mean, std):
    return data * std + mean


def fourier_transform_2d(img):
    result = torch.fft.rfft2(img)[..., :-1]
    return torch.concat([result.real, result.imag], dim=-1)


def inv_fourier_transform_2d(img):
    half_img_len = img.shape[-1] // 2
    input_tensor = img[..., :half_img_len] + img[..., half_img_len:] * 1j
    padding = torch.zeros(size=(*input_tensor.shape[:-1],1), dtype=torch.cfloat)
    input_tensor = torch.concat([input_tensor, padding], dim=-1)
    return torch.fft.irfft2(input_tensor)


def cosine_transform_2d(img):
    return torch.tensor(scipy.fft.dctn(img.numpy(), norm='ortho', axes=(-2, -1)))


def inv_cosine_transform_2d(img):
    return torch.tensor(scipy.fft.idctn(img.numpy(), norm='ortho', axes=(-2, -1)))

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


def plot_3d_tensor(curves):
    """
    Plots a 3D graph of a 2D PyTorch tensor.

    Parameters:
    - curves: A 2D PyTorch tensor of shape (y, x).
    """
    # Ensure curves is a numpy array for compatibility with matplotlib
    if isinstance(curves, torch.Tensor):
        curves = curves.numpy()

    # Generate x and y indices
    y = np.arange(curves.shape[0])
    x = np.arange(curves.shape[1])
    x, y = np.meshgrid(x, y)

    # Create a 3D plot
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Use the x, y indices as the ground plane and the tensor values as z coordinates
    z = curves

    # Plot the surface
    ax.plot_surface(x, y, z, cmap='viridis')

    # Labels for clarity
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis (Values)')

    # Show the plot
    plt.show()
    # noinspection PyTypeChecker
    plt.gcf().canvas.mpl_connect('key_press_event', _press)
    while not plt.waitforbuttonpress():
        pass
    plt.close()

    return pressed_key

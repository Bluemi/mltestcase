import numpy as np
from matplotlib import pyplot as plt


pressed_key = None


def press(event):
    global pressed_key
    pressed_key = event.key


def imshow(img):
    # img = img / 2 + 0.5
    np_img = img.numpy()
    np_img = np.minimum(np.maximum(np_img, 0.0), 1.0)

    plt.ion()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))

    plt.show()
    plt.gcf().canvas.mpl_connect('key_press_event', press)
    plt.waitforbuttonpress()
    plt.close()

    return pressed_key

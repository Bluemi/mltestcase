import numpy as np
from matplotlib import pyplot as plt


def imshow(img):
    # img = img / 2 + 0.5
    np_img = img.numpy()
    np_img = np.minimum(np.maximum(np_img, 0.0), 1.0)
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()



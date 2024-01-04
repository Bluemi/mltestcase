import numbers
from typing import Tuple
from abc import ABCMeta, abstractmethod

import numpy as np
import pygame as pg
import torch

DEFAULT_SCREEN_SIZE = (800, 600)


def gray(brightness):
    return pg.Color(brightness, brightness, brightness)


def to_np_array(p):
    if isinstance(p, np.ndarray):
        return p
    return np.array(p)


class InteractiveVisualization(metaclass=ABCMeta):
    def __init__(self, screen_size: None | Tuple[int, int] = None, framerate: int = 60):
        pg.init()
        pg.key.set_repeat(130, 25)

        screen_size = screen_size or DEFAULT_SCREEN_SIZE
        if screen_size == (0, 0):
            screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)
        else:
            screen = pg.display.set_mode(screen_size)
        self.screen = screen

        self.running = True
        self.clock = pg.time.Clock()
        self.framerate = framerate

    def run(self):
        delta_time = 0
        while self.running:
            self.handle_events()
            self.tick(delta_time)
            self.render()
            pg.display.flip()
            delta_time = self.clock.tick(self.framerate)
        pg.quit()

    @abstractmethod
    def tick(self, delta_time):
        pass

    @abstractmethod
    def render(self):
        pass

    def handle_events(self):
        events = pg.event.get()
        for event in events:
            self.handle_event(event)

    @abstractmethod
    def handle_event(self, event: pg.event.Event):
        if event.type == pg.QUIT:
            self.running = False


def create_affine_transformation(
        translation: float | np.ndarray | Tuple[int, int] = 0, scale: float | np.ndarray | Tuple[float, float] = 1
) -> np.ndarray:
    if isinstance(scale, numbers.Number):
        scale = (scale, scale)
    scale_coord = np.array(
        [[scale[0], 0, 0],
         [0, scale[1], 0],
         [0, 0, 1]]
    )
    if isinstance(translation, numbers.Number):
        translation = (translation, translation)
    translate_coord = np.array(
        [[1, 0, translation[0]],
         [0, 1, translation[1]],
         [0, 0, 1]]
    )
    return translate_coord @ scale_coord


def transform(transform_matrix: np.ndarray, mat: np.ndarray, perspective=False):
    """
    Transforms a given matrix with the given transformation matrix.
    Transformation matrix should be of shape [2, 2] or [3, 3]. If transformation matrix is of shape [3, 3] and the
    matrix to transform is of shape [2, N], matrix will be padded with ones to shape [3, N].
    If mat is of shape [2,] it will be converted to [2, 1].

    The calculation will be transform_matrix @ mat.

    :param transform_matrix: A np.ndarray with shape [2, 2] or [3, 3].
    :param mat: The matrix to convert of shape [2, N]. If mat is of shape [2,] it will be converted to [2, 1].
    :param perspective: If perspective is True and the transform_mat is of shape (3, 3), the x- and y-axis of the
                        resulting vector are divided by the resulting z axis.
    :return:
    """
    expanded = False
    if mat.shape == (2,):
        mat = mat.reshape((2, 1))
        expanded = True

    padded = False
    if transform_matrix.shape == (3, 3):
        mat = np.concatenate([mat, np.ones((1, mat.shape[1]))], axis=0)
        padded = True

    result = transform_matrix @ mat

    if expanded:
        result = result[:, 0]

    if padded:
        if perspective:
            result = result[:-1] / result[-1]
        else:
            result = result[:-1]
    return result


class CoordinateSystem:
    def __init__(self, screen_size: Tuple[int, int] | np.ndarray):
        screen_size = to_np_array(screen_size)
        coord = create_affine_transformation(screen_size/2, (100, -100))
        self.coord: np.ndarray = coord
        self.inverse_coord: np.ndarray = np.linalg.pinv(self.coord)

    def zoom_out(self, focus_point=None):
        scale = 1 / 1.2
        scale_mat = create_affine_transformation(scale=scale)
        self.coord = self.coord @ scale_mat
        if focus_point is not None:
            translation = (focus_point - self.get_zero_screen_point().flatten()) * (1 - scale)
            self.translate(translation)
        self.update_inv()

    def zoom_in(self, focus_point=None):
        scale = 1.2
        scale_mat = create_affine_transformation(scale=scale)
        self.coord = self.coord @ scale_mat
        if focus_point is not None:
            translation = (focus_point - self.get_zero_screen_point().flatten()) * (1 - scale)
            self.translate(translation)
        self.update_inv()

    def translate(self, direction):
        direction *= np.array([1, -1])
        translation_mat = create_affine_transformation(translation=direction / self.coord[0, 0])
        self.coord = self.coord @ translation_mat
        self.update_inv()

    def get_zero_screen_point(self):
        """
        Get the zero point of the coordinate system in screen coordinates.
        """
        return self.space_to_screen(np.array([0.0, 0.0]))

    def space_to_screen(self, mat: np.ndarray):
        """
        Transform the given matrix with the internal coordinates.

        :param mat: A list of column vectors with shape [2, N]. For vectors shape should be [2, 1].
        :return: A list of column vectors with shape [2, N].
        """
        mat = to_np_array(mat)
        if mat.shape == (2,):
            mat = mat.reshape(2, 1)
        return transform(self.coord, mat)

    def screen_to_space(self, mat: np.ndarray):
        mat = to_np_array(mat)
        if mat.shape == (2,):
            mat = mat.reshape(2, 1)
        return transform(self.inverse_coord, mat)

    def update_inv(self):
        self.inverse_coord = np.linalg.pinv(self.coord)


def tensor_to_pg_img(image: torch.Tensor):
    image = np.swapaxes((image * 255).numpy(), 1, 2)
    image = image.astype(np.uint8)
    image = np.moveaxis(image, 0, 2)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1).reshape(*image.shape[:2], 3)

    return pg.surfarray.make_surface(image)


class Vec2Img(InteractiveVisualization):
    def __init__(
            self, model, samples: Tuple[torch.Tensor, torch.Tensor], screen_size: None | Tuple[int, int] = None,
            framerate: int = 60
    ):
        """

        :param model: The model that can be used to transform image to vec and vice versa.
        :type model: model.MnistAutoencoder
        :param samples: A tuple [images, labels].
                        Images is a list of example images with shape [N, C, W, H], where N is the number of images, C
                        is the number of channels, W is the width and H is the height.
                        Labels is the list of corresponding labels with shape [N].
        :type samples: torch.Tensor
        :param screen_size: The start screen size of the pygame window
        :param framerate: The framerate that is used to render
        """
        super().__init__(screen_size=screen_size, framerate=framerate)
        self.model = model
        self.samples = samples
        self.sample_positions = self.calc_sample_positions()
        self.images = self.calc_images()
        self.coordinate_system = CoordinateSystem(self.screen.get_size())
        self.dragging = False
        self.mouse_position = np.zeros(2, dtype=int)

    def calc_sample_positions(self):
        with torch.no_grad():
            return self.model.encode(self.samples[0]).numpy()

    def calc_images(self):
        with torch.no_grad():
            images = [tensor_to_pg_img(i) for i in self.samples[0]]
        return images

    def tick(self, delta_time):
        pass

    def render(self):
        self.screen.fill(gray(0))

        for image, pos in zip(self.images, self.sample_positions):
            screen_pos = self.coordinate_system.space_to_screen(pos).astype(int)
            self.screen.blit(image, tuple(screen_pos.flatten()))

    def handle_event(self, event: pg.event.Event):
        super().handle_event(event)
        if event.type == pg.MOUSEBUTTONDOWN:
            self.dragging = True
        elif event.type == pg.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pg.MOUSEMOTION:
            self.mouse_position = np.array(event.pos, dtype=int)
            if self.dragging:
                self.coordinate_system.translate(np.array(event.rel))
        elif event.type == pg.MOUSEWHEEL:
            if event.y < 0:
                self.coordinate_system.zoom_out(focus_point=self.mouse_position)
            else:
                self.coordinate_system.zoom_in(focus_point=self.mouse_position)

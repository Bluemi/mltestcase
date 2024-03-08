import numbers
from typing import Tuple
from abc import ABCMeta, abstractmethod

import numpy as np
import pygame as pg
import torch

from utils import denormalize

DEFAULT_SCREEN_SIZE = (800, 600)
COLORS = np.array([
    [0.12, 0.47, 0.71],
    [1.00, 0.50, 0.05],
    [0.17, 0.63, 0.17],
    [0.84, 0.15, 0.16],
    [0.58, 0.40, 0.74],
    [0.55, 0.34, 0.29],
    [0.89, 0.47, 0.76],
    [0.50, 0.50, 0.50],
    [0.74, 0.74, 0.13],
    [0.09, 0.75, 0.81],
])


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
        self.render_needed = True
        self.clock = pg.time.Clock()
        self.framerate = framerate

    def run(self):
        delta_time = 0
        while self.running:
            self.handle_events()
            self.tick(delta_time)
            if self.render_needed:
                self.render()
                self.render_needed = False
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
        elif event.type in (pg.WINDOWSIZECHANGED, pg.KEYUP, pg.ACTIVEEVENT):
            self.render_needed = True


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

        # user control
        self.dragging: bool = False
        self.mouse_position = np.zeros(2, dtype=int)

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

    def handle_event(self, event):
        render_needed = False
        if event.type == pg.MOUSEBUTTONDOWN:
            self.dragging = True
            render_needed = True
        elif event.type == pg.MOUSEBUTTONUP:
            self.dragging = False
            render_needed = True
        elif event.type == pg.MOUSEMOTION:
            self.mouse_position = np.array(event.pos, dtype=int)
            if self.dragging:
                self.translate(np.array(event.rel))
                render_needed = True
        elif event.type == pg.MOUSEWHEEL:
            if event.y < 0:
                self.zoom_out(focus_point=self.mouse_position)
                render_needed = True
            else:
                self.zoom_in(focus_point=self.mouse_position)
                render_needed = True
        return render_needed, self


def tensor_to_pg_img(image: torch.Tensor, alpha_threshold=0, color=None, normalization_mean_std=(0, 1)):
    image = denormalize(image, normalization_mean_std[0], normalization_mean_std[1])
    image = np.minimum(np.maximum(image, 0), 1)

    image = np.swapaxes((image * 255).numpy(), 1, 2)
    image = np.moveaxis(image, 0, 2)

    alpha_mask = (image > alpha_threshold).astype(np.uint8) * 255
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1).reshape(*image.shape[:2], 3)

    # handle color
    if color is not None:
        image *= color.reshape(1, 1, -1)

    surface = pg.surfarray.make_surface(image.astype(int))

    # add alpha
    surface = surface.convert_alpha()
    alpha_pixels = pg.surfarray.pixels_alpha(surface)
    alpha_pixels[:] = alpha_mask.reshape(alpha_mask.shape[:2])

    return surface


def get_raster_coordinates(min_coord, max_coord, num_points_wanted):
    def adapt_quotient(quotient):
        target_dividends = np.exp(np.linspace(0, np.log(10), 10))
        if quotient <= 0:
            raise ValueError('Invalid quotient: {}'.format(quotient))
        numb_ten_potency = 0
        while quotient > 10:
            quotient *= 0.1
            numb_ten_potency += 1
        while quotient < 1:
            quotient *= 10
            numb_ten_potency -= 1

        diffs = np.abs(quotient - target_dividends)
        index = np.argmin(diffs)
        best_fitting = target_dividends[index] * (10 ** numb_ten_potency)

        return best_fitting

    width = max_coord - min_coord
    space_between_points = adapt_quotient(width / num_points_wanted)
    coord_minimum = np.round(min_coord / space_between_points) * space_between_points
    coord_maximum = np.round(max_coord / space_between_points) * space_between_points + space_between_points

    return np.arange(coord_minimum, coord_maximum + space_between_points, space_between_points)



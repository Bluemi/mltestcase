import enum
from typing import Tuple

import numpy as np
import pygame as pg
import torch

from utils import fourier_transform_2d, cosine_transform_2d, inv_fourier_transform_2d, inv_cosine_transform_2d
from utils.interactive_visualizations import InteractiveVisualization, CoordinateSystem, COLORS, tensor_to_pg_img, gray, \
    get_raster_coordinates


class Vec2Img(InteractiveVisualization):
    class RenderMode(enum.Enum):
        ENCODING = 0
        DECODING = 1

        def next(self):
            if self == self.ENCODING:
                return self.DECODING
            elif self == self.DECODING:
                return self.ENCODING

    def __init__(
            self, model, samples: Tuple[torch.Tensor, torch.Tensor], screen_size: None | Tuple[int, int] = None,
            framerate: int = 60, normalization_mean_std=(0, 1), use_ft=False,
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
        :param normalization_mean_std: Tuple of mean and std used to denormalize the images.
        :param use_ft: Whether to apply fft on images before feeding to the model
        """
        super().__init__(screen_size=screen_size, framerate=framerate)
        self.model = model
        self.samples = samples
        self.normalization_mean_std = normalization_mean_std
        self.use_ft = use_ft
        self.sample_positions = self.calc_sample_positions()
        self.images = self.calc_images(color_images=False, labels_from_model=False)
        self.colored_images = self.calc_images(color_images=True, labels_from_model=False)
        self.colored_model_images = self.calc_images(color_images=True, labels_from_model=True)
        self.coordinate_system = CoordinateSystem(self.screen.get_size())
        self.dragging = False
        self.show_color_mode = 0
        self.render_mode = Vec2Img.RenderMode.ENCODING
        self.mouse_position = np.zeros(2, dtype=int)

    def calc_sample_positions(self):
        with torch.no_grad():
            inputs = self.samples[0]
            if self.use_ft == 'fft':
                inputs = fourier_transform_2d(self.samples[0])
            if self.use_ft == 'dct':
                inputs = cosine_transform_2d(self.samples[0])

            return self.model.encode(inputs).numpy()

    def calc_images(self, color_images=False, labels_from_model=False):
        with torch.no_grad():
            images = []
            for img, label in zip(self.samples[0], self.samples[1]):
                if color_images:
                    if labels_from_model:
                        model_input = img
                        if self.use_ft == 'fft':
                            model_input = fourier_transform_2d(img)
                        if self.use_ft == 'dct':
                            model_input = cosine_transform_2d(img)
                        label = torch.argmax(self.model.forward_classify(model_input)).item()
                    color = COLORS[label]
                    img = tensor_to_pg_img(img, 128, color, normalization_mean_std=self.normalization_mean_std)
                else:
                    img = tensor_to_pg_img(img, 128, normalization_mean_std=self.normalization_mean_std)
                images.append(img)
        return images

    def tick(self, delta_time):
        pass

    def render(self):
        self.screen.fill(gray(0))

        if self.render_mode == Vec2Img.RenderMode.ENCODING:
            if self.show_color_mode == 0:
                images = self.colored_images
            elif self.show_color_mode == 1:
                images = self.colored_model_images
            elif self.show_color_mode == 2:
                images = self.images
            else:
                raise ValueError("unknown show color mode: {}".format(self.show_color_mode))

            for image, pos in zip(images, self.sample_positions):
                screen_pos = self.coordinate_system.space_to_screen(pos).astype(int)
                self.screen.blit(image, tuple(screen_pos.flatten()))
        elif self.render_mode == Vec2Img.RenderMode.DECODING:
            self.render_decoding()

    def render_decoding(self):
        screen_size = self.screen.get_size()
        extreme_points_screen = np.array([[0, 0], screen_size])
        extreme_points_space = self.coordinate_system.screen_to_space(extreme_points_screen.T).T

        y_raster = get_raster_coordinates(extreme_points_space[1, 1], extreme_points_space[0, 1], screen_size[1] // 40)
        x_raster = get_raster_coordinates(extreme_points_space[0, 0], extreme_points_space[1, 0], screen_size[0] // 40)
        grid = np.meshgrid(x_raster, y_raster, indexing='xy')
        grid = np.stack(grid, axis=2).reshape(-1, 2)

        grid_tensor = torch.tensor(grid, dtype=torch.float32)
        with torch.no_grad():
            decoded_images = self.model.decode(grid_tensor).reshape(-1, 1, 28, 28)
            if self.use_ft == 'fft':
                decoded_images = inv_fourier_transform_2d(decoded_images)
            elif self.use_ft == 'dct':
                decoded_images = inv_cosine_transform_2d(decoded_images)
            labels = torch.argmax(self.model.classification_head(grid_tensor), dim=1)

        for pos, image, label in zip(grid, decoded_images, labels):
            screen_pos = self.coordinate_system.space_to_screen(pos)
            color = None
            if self.show_color_mode in (0, 1):
                color = COLORS[label]
            img = tensor_to_pg_img(
                image, 32, color=color, normalization_mean_std=self.normalization_mean_std
            )
            self.screen.blit(img, tuple(screen_pos.flatten()))

    def handle_event(self, event: pg.event.Event):
        super().handle_event(event)
        if self.coordinate_system.handle_event(event):
            self.render_needed = True

        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_c:
                if pg.key.get_mods() & pg.KMOD_SHIFT:
                    self.show_color_mode = (self.show_color_mode - 1) % 3
                else:
                    self.show_color_mode = (self.show_color_mode + 1) % 3
                self.render_needed = True
            if event.key == pg.K_e:
                if pg.key.get_mods() & pg.KMOD_SHIFT:
                    self.render_mode = self.RenderMode.DECODING
                else:
                    self.render_mode = self.RenderMode.ENCODING
                self.render_needed = True
        elif event.type in (pg.WINDOWSIZECHANGED, pg.KEYUP, pg.ACTIVEEVENT):
            self.render_needed = True

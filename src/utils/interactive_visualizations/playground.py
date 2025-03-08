import enum
from typing import Tuple, List

import torch
import numpy as np
import pygame as pg
from torchsummary import summary

from model.playground import PlaygroundModel
from utils.datasets.playground import get_playground_dataloader
from utils.interactive_visualizations import InteractiveVisualization, CoordinateSystem, gray


class DataKind(enum.Enum):
    CIRCLE = 0
    XOR = 1
    LINEAR = 2
    SPIRALS = 3

    @classmethod
    def next(cls, current):
        members: List = list(cls)
        index = members.index(current)
        return members[(index + 1) % len(members)]


class Playground(InteractiveVisualization):
    def __init__(self, screen_size: None | Tuple[int, int] = None, num_data_points: int = 1024):
        """
        :param screen_size: The start screen size of the pygame window
        """
        super().__init__(screen_size=screen_size, framerate=60)
        self.coordinate_system = CoordinateSystem(self.screen.get_size())

        self.data_kind = DataKind.CIRCLE
        self.num_data_points = num_data_points
        self.points, self.labels = generate_data(self.data_kind, self.num_data_points)

        self.model = PlaygroundModel(activation_function='moth')
        summary(self.model, (2,), device='cpu')
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.002, weight_decay=0.0, momentum=0.9)
        self.dataset = get_playground_dataloader(self.points, self.labels, batch_size=128, shuffle=True)

    def tick(self, delta_time):
        for batch in self.dataset:
            inputs, labels = batch
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)

            loss.backward()
            self.optimizer.step()

            # print('loss:', loss.item())
        self.render_needed = True

    def render(self):
        self.screen.fill(gray(0))

        self._render_prediction()
        self._render_data()

    def _render_data(self):
        render_positions = self.coordinate_system.space_to_screen(self.points.T).T
        for render_position, label in zip(render_positions, self.labels):
            color_point = pg.Color(255, 128, 0) if label == 1 else pg.Color(0, 180, 255)
            color_border = pg.Color(255, 228, 180) if label == 1 else pg.Color(80, 240, 255)
            pg.draw.circle(self.screen, color_border, render_position, 3)
            pg.draw.circle(self.screen, color_point, render_position, 2)

    def _render_prediction(self):
        screen_size = self.screen.get_size()
        extreme_points_screen = np.array([[0, 0], screen_size])
        extreme_points_space = self.coordinate_system.screen_to_space(extreme_points_screen.T).T

        pixel_size = 8
        y_raster = np.linspace(extreme_points_space[1, 1], extreme_points_space[0, 1], screen_size[1] // pixel_size)
        x_raster = np.linspace(extreme_points_space[0, 0], extreme_points_space[1, 0], screen_size[0] // pixel_size)
        grid = np.meshgrid(x_raster, y_raster, indexing='xy')
        grid = np.stack(grid, axis=2)
        grid_shape = grid.shape
        grid = grid.reshape(-1, 2)

        grid_tensor = torch.tensor(grid, dtype=torch.float32, requires_grad=False)
        with torch.no_grad():
            result = self.model(grid_tensor)
            colors = interpolate_colors(result)
            colors = colors.reshape(grid_shape[0], grid_shape[1], 3)
            colors = torch.swapaxes(colors, 0, 1)
            # noinspection PyTypeChecker
            img = colors.to(int).numpy()

            img = pg.surfarray.make_surface(img[:, ::-1])
            pg.transform.smoothscale(img, (screen_size[0], screen_size[1]), self.screen)

    def handle_event(self, event: pg.event.Event):
        super().handle_event(event)
        if self.coordinate_system.handle_event(event):
            self.render_needed = True

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_d:
                self.data_kind = DataKind.next(self.data_kind)
                self.points, self.labels = generate_data(self.data_kind, self.num_data_points)
                self.dataset = get_playground_dataloader(self.points, self.labels, 32, True)
                self.render_needed = True


def generate_data(data_kind, num_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    if data_kind == DataKind.CIRCLE:
        space = np.random.uniform(low=0, high=2*np.pi, size=num_points//2)
        outer_points = np.concatenate([np.sin(space).reshape(-1, 1), np.cos(space).reshape(-1, 1)], axis=1)
        outer_points += np.random.normal(0, 0.1, size=(num_points//2, 2))
        inner_points = np.random.normal(0, 0.2, size=(num_points//2, 2))
        points = np.concatenate([outer_points, inner_points], axis=0)

        labels = np.zeros(points.shape[0]) - 1
        labels[num_points//2:] = 1
    elif data_kind == DataKind.XOR:
        points = np.random.uniform(-1, 1, size=(num_points, 2))
        labels = np.equal(points[:, 0] > 0, points[:, 1] > 0) * 2 - 1
    elif data_kind == DataKind.LINEAR:
        left_points: np.ndarray = np.random.normal(loc=(0.2, 0.7), scale=0.25, size=(num_points//2, 2))
        right_points: np.ndarray = np.random.normal(loc=(-0.7, -0.2), scale=0.25, size=(num_points//2, 2))
        points = np.concatenate([left_points, right_points], axis=0)
        labels = np.zeros(points.shape[0]) - 1
        labels[num_points//2:] = 1
    elif data_kind == DataKind.SPIRALS:
        outer_space = np.random.uniform(low=0, high=2 * np.pi, size=num_points // 2)
        outer_points = np.concatenate(
            [
                (np.sin(outer_space*2) * outer_space * 0.2).reshape(-1, 1),
                (np.cos(outer_space*2) * outer_space * 0.2).reshape(-1, 1)
            ],
            axis=1
        )

        inner_space = np.random.uniform(low=0, high=2 * np.pi, size=num_points // 2)
        inner_points = np.concatenate(
            [
                (np.sin(inner_space*2 + np.pi) * inner_space * 0.2).reshape(-1, 1),
                (np.cos(inner_space*2 + np.pi) * inner_space * 0.2).reshape(-1, 1)
            ],
            axis=1
        )

        points = np.concatenate([outer_points, inner_points], axis=0)
        points += np.random.normal(0, 0.01, size=(points.shape[0], 2))

        labels = np.zeros(points.shape[0]) - 1
        labels[num_points // 2:] = 1
    else:
        raise ValueError('Unknown data_kind: {}'.format(data_kind.name))
    return points, labels


def interpolate_colors(values):
    values = torch.clip(values, -1, 1)

    orange = torch.tensor([[255, 168, 40]], dtype=torch.float32)
    white = torch.tensor([[255, 255, 255]], dtype=torch.float32)
    cyan = torch.tensor([[40, 220, 255]], dtype=torch.float32)

    # interpolate for cyan case
    cyan_colors = -values[:, None] * cyan + ((1+values[:, None]) * white)

    # interpolate for orange case
    orange_colors = values[:, None] * orange + ((1-values[:, None]) * white)

    positive_indices = np.where(values > 0)
    cyan_colors[positive_indices] = orange_colors[positive_indices]

    return cyan_colors

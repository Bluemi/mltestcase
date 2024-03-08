import enum
from typing import Tuple

import numpy as np
import pygame as pg

from utils import describe
from utils.interactive_visualizations import InteractiveVisualization, CoordinateSystem, gray


class DataKind(enum.Enum):
    CIRCLE = 0
    XOR = 1
    LINEAR = 2
    SPIRALS = 3

    @classmethod
    def next(cls, current):
        members = list(cls)
        index = members.index(current)
        return members[(index + 1) % len(members)]


class Playground(InteractiveVisualization):
    def __init__(self, screen_size: None | Tuple[int, int] = None):
        """
        :param screen_size: The start screen size of the pygame window
        """
        super().__init__(screen_size=screen_size, framerate=60)
        self.coordinate_system = CoordinateSystem(self.screen.get_size())

        self.data_kind = DataKind.CIRCLE
        self.points, self.labels = Playground.generate_data(self.data_kind)

        describe(self.points, 'points')

    @staticmethod
    def generate_data(data_kind, num_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        if data_kind == DataKind.CIRCLE:
            space = np.random.uniform(low=0, high=2*np.pi, size=num_points//2)
            outer_points = np.concatenate([np.sin(space).reshape(-1, 1), np.cos(space).reshape(-1, 1)], axis=1)
            outer_points += np.random.normal(0, 0.1, size=(num_points//2, 2))
            inner_points = np.random.normal(0, 0.2, size=(num_points//2, 2))
            points = np.concatenate([outer_points, inner_points], axis=0)

            labels = np.zeros(points.shape[0])
            labels[num_points//2:] = 1
        elif data_kind == DataKind.XOR:
            points = np.random.uniform(-1, 1, size=(num_points, 2))
            labels = np.equal(points[:, 0] > 0, points[:, 1] > 0)
        elif data_kind == DataKind.LINEAR:
            left_points: np.ndarray = np.random.normal(loc=(0.2, 0.7), scale=0.25, size=(num_points//2, 2))
            right_points: np.ndarray = np.random.normal(loc=(-0.7, -0.2), scale=0.25, size=(num_points//2, 2))
            points = np.concatenate([left_points, right_points], axis=0)
            labels = np.zeros(points.shape[0])
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

            labels = np.zeros(points.shape[0])
            labels[num_points // 2:] = 1
        else:
            raise ValueError('Unknown data_kind: {}'.format(data_kind.name))
        return points, labels

    def tick(self, delta_time):
        pass

    def render(self):
        self.screen.fill(gray(0))

        render_positions = self.coordinate_system.space_to_screen(self.points.T).T

        for render_position, label in zip(render_positions, self.labels):
            color = pg.Color(255, 128, 0) if label == 1 else pg.Color(0, 180, 255)
            pg.draw.circle(self.screen, color, render_position, 3)

    def handle_event(self, event: pg.event.Event):
        super().handle_event(event)
        if self.coordinate_system.handle_event(event):
            self.render_needed = True

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_d:
                self.data_kind = DataKind.next(self.data_kind)
                self.points, self.labels = self.generate_data(self.data_kind)
                self.render_needed = True

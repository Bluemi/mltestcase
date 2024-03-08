from typing import Tuple

import numpy as np
import pygame as pg

from utils.interactive_visualizations import InteractiveVisualization, CoordinateSystem, gray


class Playground(InteractiveVisualization):
    def __init__(self, screen_size: None | Tuple[int, int] = None):
        """
        :param screen_size: The start screen size of the pygame window
        """
        super().__init__(screen_size=screen_size, framerate=0)
        self.coordinate_system = CoordinateSystem(self.screen.get_size())

        self.points = np.random.normal(size=(10, 2))

    def tick(self, delta_time):
        pass

    def render(self):
        self.screen.fill(gray(0))

        render_positions = self.coordinate_system.space_to_screen(self.points.T).T

        for render_position in render_positions:
            pg.draw.circle(self.screen, gray(255), render_position, 3)

    def handle_event(self, event: pg.event.Event):
        super().handle_event(event)
        if self.coordinate_system.handle_event(event):
            self.render_needed = True

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_c:
                self.render_needed = True
        elif event.type in (pg.WINDOWSIZECHANGED, pg.KEYUP, pg.ACTIVEEVENT):
            self.render_needed = True

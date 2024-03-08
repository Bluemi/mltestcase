import enum
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
        self.dragging = False
        self.mouse_position = np.zeros(2, dtype=int)

        self.points = None

    def tick(self, delta_time):
        pass

    def render(self):
        self.screen.fill(gray(0))

        # TODO

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
                self.render_needed = True
        elif event.type == pg.MOUSEWHEEL:
            if event.y < 0:
                self.coordinate_system.zoom_out(focus_point=self.mouse_position)
            else:
                self.coordinate_system.zoom_in(focus_point=self.mouse_position)
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

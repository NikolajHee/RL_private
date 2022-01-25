# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import time
import numpy as np
import pyglet
import pyglet.shapes
from pyglet import shapes
from pyglet.graphics import OrderedGroup
from pyglet.shapes import Rectangle, Line, Sector, Circle
from pyglet.text import Label
from scipy import interpolate
from irlc.pacman.pacman_resources import BLACK, WHITE, PACMAN_COLOR, Pacman
from irlc.pacman.pacman_resources import Ghost
from irlc.utils.pyglet_rendering import PygletViewer

MUTED_TEXT_COLOR = (int(0.7 * 255),) * 3
OBSTACLE_COLOR = (int(255 * 0.5),) * 3


def format_color(r, g, b):
    return int(r * 255), int(g * 255), int(b * 255)


DEFAULT_GRID_SIZE = 30.0
INFO_PANE_HEIGHT = 35

BACKGROUND_COLOR = BLACK
WALL_COLOR = format_color(0.0 / 255.0, 51.0 / 255.0, 255.0 / 255.0)
INFO_PANE_COLOR = format_color(.4, .4, 0)
SCORE_COLOR = format_color(.9, .9, .9)
PACMAN_OUTLINE_WIDTH = 2
PACMAN_CAPTURE_OUTLINE_WIDTH = 4

GHOST_COLORS = [format_color(.9, 0, 0), format_color(0, .3, .9), format_color(.98, .41, .07), format_color(.1, .75, .7),
                format_color(1.0, 0.6, 0.0), format_color(.4, 0.13, 0.91)]

GHOST_SIZE = 0.65
SCARED_COLOR = WHITE

# GHOST_VEC_COLORS = [colorToVector(gc) for gc in GHOST_COLORS]
PACMAN_SCALE = 0.5

# Food
FOOD_COLOR = WHITE
FOOD_SIZE = 0.1

# Capsule graphics
CAPSULE_COLOR = WHITE
CAPSULE_SIZE = 0.25
# Drawing walls
WALL_RADIUS = 0.15


class PacmanViewer(PygletViewer):
    plan = []
    plan_batch = None

    def __init__(self, data, zoom=1.0, frame_time=0.05):
        self.data = data
        self.gridSize = DEFAULT_GRID_SIZE * zoom

        self.frame_time = frame_time
        width = data.layout.width
        height = data.layout.height

        grid_width = (width - 1) * self.gridSize
        grid_height = (height - 1) * self.gridSize
        screen_width = 2 * self.gridSize + grid_width
        screen_height = 2 * self.gridSize + grid_height + INFO_PANE_HEIGHT

        self.food = {}
        self.capsules = {}
        batch = pyglet.graphics.Batch()
        agent_batch = pyglet.graphics.Batch()

        food_group = OrderedGroup(10)

        super().__init__(screen_width=int(screen_width), xmin=0, xmax=screen_width, ymin=0, ymax=screen_height)

        self.bg = shapes.Rectangle(0, 0, screen_width, screen_height, color=BLACK, batch=batch, group=OrderedGroup(-1))

        for i, j in data.food.asList():
            self.food[i, j] = Circle((i + 1) * self.gridSize, (j + 1) * self.gridSize + INFO_PANE_HEIGHT,
                                     FOOD_SIZE * self.gridSize, color=FOOD_COLOR, batch=batch, group=food_group)

        for i, j in data.capsules:
            self.capsules[i, j] = Circle((i + 1) * self.gridSize, (j + 1) * self.gridSize + INFO_PANE_HEIGHT,
                                         CAPSULE_SIZE * self.gridSize, color=CAPSULE_COLOR, batch=batch,
                                         group=food_group)

        self.dx = self.gridSize
        self.dy = self.gridSize + INFO_PANE_HEIGHT

        self.agents = [
            Ghost(batch=agent_batch, agent_index=k, order=1000 + k * 3) if k > 0 else Pacman(grid_size=self.gridSize,
                                                                                             order=1000,
                                                                                             batch=agent_batch) for k in
            range(len(data.agentStates))]
        for j in range(1, len(self.agents)):
            self.agents[j].group.scale(self.gridSize * GHOST_SIZE)

        # Now draw the beliefs.
        self.tiles = {}
        g0 = OrderedGroup(0)
        for i in range(data.layout.width):
            for j in range(data.layout.height):
                if not data.layout.walls.data[i][j]:
                    x, y = i * self.gridSize + self.dx, j * self.gridSize + self.dy
                    self.tiles[i, j] = Rectangle(x - self.gridSize / 2, y - self.gridSize / 2, self.gridSize,
                                                 self.gridSize, color=BLACK, batch=batch, group=g0)

        # Now make the walls.
        self._make_walls(data, batch)
        self.batch = batch
        self.agent_batch = agent_batch

        text = "SCORE: % 4d" % 0
        self.score_label = Label(text, x=self.gridSize, y=INFO_PANE_HEIGHT, font_size=24, color=PACMAN_COLOR + (255,),
                                 font_name='Arial', bold=True, batch=batch, anchor_y='center')

        size = 10 if screen_width < 160 else (12 if screen_width < 240 else 20)
        self.ghostDistanceText = [Label('', x=screen_width / 2 + screen_width / 8 * i, y=self.gridSize / 2,
                                        font_size=size, color=GHOST_COLORS[i + 1] + (255,), font_name='Arial',
                                        bold=True, batch=batch, anchor_y='center') for i in range(4)]

    def animate_pacman(self, old_state, new_state, frames=4):
        fx, fy = old_state.agentStates[0].getPosition()
        px, py = new_state.agentStates[0].getPosition()
        for nframe in range(1, int(frames) + 1):
            self.agents[0].set_animation(nframe, frames + 1)
            self.agents[0].set_direction(new_state.agentStates[0].getDirection())
            pos = px * nframe / frames + fx * (frames - nframe) / frames, py * nframe / frames + fy * (
                        frames - nframe) / frames
            self.agents[0].group.translate(pos[0] * self.gridSize + self.dx, pos[1] * self.gridSize + self.dy)
            time.sleep(self.frame_time / frames)
            self.viewer.render()
        self.agents[0].set_animation(0, 4)

    def update(self, data, ghostbeliefs=None, path=None, visitedlist=None):
        for k, ag in enumerate(data.agentStates):
            i, j = ag.getPosition()
            self.agents[k].group.translate(i * self.gridSize + self.dx, j * self.gridSize + self.dy)
            self.agents[k].set_direction(ag.getDirection())
            if k > 0:
                self.agents[k].set_scared(ag.scaredTimer > 0)

        for ij in self.food:
            self.food[ij].visible = ij in data.food.asList()

        for ij in self.capsules:
            self.capsules[ij].visible = ij in data.capsules

        # Update expanded cells.
        if visitedlist is not None:
            n = len(visitedlist)
            for k, (i, j) in enumerate(visitedlist):
                self.tiles[i, j].color = tuple(int(((n - k) * c * .5 / n + .25) * 255) for c in [1.0, 0.0, 0.0])

        if path is not None:
            self._draw_path(path)

        if ghostbeliefs is not None:
            if 'ghostDistances' in dir(data):
                for k, l in enumerate(data.ghostDistances):
                    self.ghostDistanceText[k].text = str(l)

            ghostbeliefs = [gb.copy() for gb in ghostbeliefs]  # Required. Uses a default dict.
            if ghostbeliefs is None or len(ghostbeliefs) == 0:
                return
            for x in range(data.layout.walls.width):
                for y in range(data.layout.walls.height):
                    weights = [gb[x, y] for gb in ghostbeliefs]
                    color = [0.0, 0.0, 0.0]
                    colors = [(r / 255, g / 255, b / 255) for r, g, b in GHOST_COLORS[1:]]

                    for weight, gcolor in zip(weights, colors):
                        color = [min(1.0, c + 0.95 * g * weight ** .3) for c, g in zip(color, gcolor)]

                    color = (int(c * 255) for c in color)
                    if (x, y) in self.tiles:
                        self.tiles[x, y].color = color

        self.score_label.text = "SCORE: % 4d" % data.score

    def draw(self):
        self.batch.draw()
        self.agent_batch.draw()

    def _draw_path(self, path):
        color = (int(0.5 * 255), int(.95 * 255), int(0.5 * 255))
        self.plan = []
        # Update the path.
        bg = dict(batch=self.batch, group=OrderedGroup(1))
        xy = np.stack([np.asarray([x * self.gridSize + self.dx, y * self.gridSize + self.dy]) for x, y in path])

        n = len(xy)
        T = np.asarray(range(n))
        t_new = np.linspace(0, n - 1, n * 8)

        x_new = interpolate.splev(t_new, interpolate.splrep(T, xy[:, 0], s=0), der=0)
        y_new = interpolate.splev(t_new, interpolate.splrep(T, xy[:, 1], s=0), der=0)

        for i in range(len(x_new) - 1):
            self.plan.append(Line(x_new[i], y_new[i], x_new[i + 1], y_new[i + 1], color=color, width=4, **bg))

    def _make_walls(self, data, batch):
        self.capture = False

        bg = dict(batch=batch, group=OrderedGroup(1))
        self.walls = []

        def wl(p1, p2, yellow=False):
            self.walls.append(Line(*p1, *p2, color=WALL_COLOR if not yellow else PACMAN_COLOR, width=2, **bg))

        def sec(q, radius, arch):
            a1, a2 = arch[0] / 180 * np.pi + np.pi / 2, arch[1] / 180 * np.pi + np.pi / 2
            self.walls.append(Sector(q[0], q[1], radius, start_angle=a1, angle=a2 - a1, color=WALL_COLOR, **bg))
            self.walls.append(Sector(q[0], q[1], radius - 2, start_angle=a1, angle=a2 - a1, color=BLACK, **bg))

        def is_wall(x, y):
            return 0 <= x < data.layout.walls.width and 0 <= y < data.layout.walls.height and data.layout.walls[x][y]

        for xNum, wall in enumerate(data.layout.walls):
            for yNum, cell in enumerate(wall):
                if not cell:
                    continue
                p = ((xNum + 1) * self.gridSize, (yNum + 1) * self.gridSize + INFO_PANE_HEIGHT)

                # draw each quadrant of the square based on adjacent walls
                w_is_wall = is_wall(xNum - 1, yNum)
                e_is_wall = is_wall(xNum + 1, yNum)
                n_is_wall = is_wall(xNum, yNum + 1)
                s_is_wall = is_wall(xNum, yNum - 1)
                nw_is_wall = is_wall(xNum - 1, yNum + 1)
                sw_is_wall = is_wall(xNum - 1, yNum - 1)
                ne_is_wall = is_wall(xNum + 1, yNum + 1)
                se_is_wall = is_wall(xNum + 1, yNum - 1)

                r = WALL_RADIUS * self.gridSize
                if not n_is_wall and not e_is_wall:
                    sec(p, r, (0 - 90, 91 - 90))
                if n_is_wall and not e_is_wall:
                    wl(add(p, (r, 0)), add(p, (r, self.gridSize / 2 - 1)))
                if not n_is_wall and e_is_wall:
                    wl(add(p, (0, r)), add(p, (self.gridSize * 0.5 + 1, r)))
                if n_is_wall and e_is_wall and (not ne_is_wall):
                    sec(add(p, (2 * r, 2 * r)), r - 1, (180 - 90, 271 - 90))
                    wl(add(p, (2 * r - 1, r)), add(p, (self.gridSize * 0.5 + 1, r)))
                    wl(add(p, (r, 2 * r + 1)), add(p, (r, self.gridSize / 2)))
                if not n_is_wall and not w_is_wall:
                    sec(p, r, (90 + 180 + 90, 181 + 180 + 90))
                if n_is_wall and not w_is_wall:
                    wl(add(p, (-r, 0)), add(p, (-r, self.gridSize / 2)))
                if not n_is_wall and w_is_wall:
                    wl(add(p, (0, r)), add(p, (self.gridSize * (-0.5), r)))
                if n_is_wall and w_is_wall and not nw_is_wall:
                    sec(add(p, (-2 * r, 2 * r)), r - 1, (270 - 180 + 90, 361 - 180 + 90))
                    wl(add(p, (-2 * r + 1, r)), add(p, (self.gridSize * (-0.5), r)))
                    wl(add(p, (-r, 2 * r + 1)), add(p, (-r, self.gridSize / 2)))
                if not s_is_wall and not e_is_wall:
                    sec(p, r, (180, 271))
                if s_is_wall and not e_is_wall:
                    wl(add(p, (r, 0)), add(p, (r, self.gridSize * (-0.5) + 0)))
                if not s_is_wall and e_is_wall:
                    wl(add(p, (0, -r)), add(p, (self.gridSize * 0.5 + 1, -r)))
                if s_is_wall and e_is_wall and not se_is_wall:
                    sec(add(p, (2 * r, -2 * r)), r - 1, (90 - 90, 181 - 90))
                    wl(add(p, (2 * r - 1, -r)), add(p, (self.gridSize * 0.5, -r)))
                    wl(add(p, (r, -2 * r - 0)), add(p, (r, -self.gridSize / 2)))
                if not s_is_wall and not w_is_wall:
                    sec(p, r, (180 - 90, 271 - 90))
                if s_is_wall and (not w_is_wall):
                    wl(add(p, (-r, 0)), add(p, (-r, self.gridSize * (-0.5) + 0)))
                if (not s_is_wall) and w_is_wall:
                    wl(add(p, (0, -r)), add(p, (self.gridSize * (-0.5) - 1, -r)))
                if s_is_wall and w_is_wall and (not sw_is_wall):
                    sec(add(p, (-2 * r, -2 * r)), r - 1, (0 + 90 * 3, 91 + 90 * 3))
                    wl(add(p, (-2 * r, -r)), add(p, (self.gridSize * (-0.5), -r)))
                    wl(add(p, (-r, -2 * r)), add(p, (-r, -self.gridSize / 2)))


def add(x, y):
    return x[0] + y[0], x[1] + y[1]
# 353

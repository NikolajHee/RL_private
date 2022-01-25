# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import pyglet
from pyglet import shapes
import numpy as np
from irlc.utils.pyglet_rendering import PygletViewer, GroupedElement, CameraGroup
from irlc.pacman.pacman_resources import BLACK, WHITE, Pacman
import pyglet.shapes
from pyglet.graphics import OrderedGroup
from pyglet.shapes import Rectangle, Line, Triangle
from pyglet.text import Label

MUTED_TEXT_COLOR = (int(0.7*255),)*3
OBSTACLE_COLOR = (int(255*0.5),)*3


class GridworldViewer(PygletViewer):
    def __init__(self, mdp, grid_size):
        self.mdp = mdp
        self.GRID_SIZE = grid_size
        self.MARGIN = self.GRID_SIZE * 0.75
        screen_width = (mdp.width - 1) * self.GRID_SIZE + self.MARGIN * 2
        screen_height = (mdp.height - 0.5) * self.GRID_SIZE + self.MARGIN * 2

        bgg = OrderedGroup(-1)
        batch = pyglet.graphics.Batch()

        super().__init__(screen_width=int(screen_width), xmin=0, xmax=screen_width, ymin=0, ymax=screen_height)
        self.bg = [shapes.Rectangle(0, 0, screen_width, screen_height, color=BLACK, batch=batch, group=bgg)]

        self.grid = {}
        for i in range(mdp.width):
            for j in range(mdp.height):
                tile = mdp.grid[(i,j)]
                tile = ' ' if tile == 'S' else tile
                square = BlankSquare(type=tile if tile in [' ', '#'] else 'E', grid_size=self.GRID_SIZE, batch=batch, order=2+j + i * self.mdp.height)
                square.group.translate(self.MARGIN + self.GRID_SIZE * (i+0), self.MARGIN + self.GRID_SIZE * (j+.5))
                self.grid[(i,j)] = square

        self.pacman = Pacman(grid_size=self.GRID_SIZE, batch=batch, order=1000)
        self.pacman.group.scale(0.3)
        self.pacman.group.translate(self.MARGIN + self.GRID_SIZE * (0+0), self.MARGIN + self.GRID_SIZE * (0+.5))

        x, y = (mdp.width - 1.0) / 2.0 * self.GRID_SIZE + self.MARGIN, self.GRID_SIZE*0.4
        self.message = Label("", x=x, y=y, color=WHITE + (255,), font_name="Ariel", font_size=32*0.75, bold=True,
                             anchor_x='center', anchor_y='center', group=OrderedGroup(1), batch=batch)
        self.batch = batch
        self.dy = self.MARGIN + self.GRID_SIZE/2

    def update_null(self, state, message=""):
        self.message.text = message
        self.set_state(state)
        for state, g in self.grid.items():
            if self.mdp.grid[state] != "#":
                g.center_text.text = str(self.mdp.grid[state]) if self.mdp.grid[state] != 'S' else ''
                for a, al in g.action_labels.items():
                    al['action'].visible = False
                    al['label'].text = ""
                    al['triangle'].color = BLACK
                for c in g.cross:
                    c.visible = False

    def update_v(self, v, state, preferred_actions=None, message=""):
        self.message.text = message
        self.set_state(state)

        min_value, max_value = min(v.values()), max(v.values())

        for state, grid in self.grid.items():
            tile = self.mdp.grid[state]
            if tile != '#':
                actions = None if preferred_actions is None else preferred_actions[state]
                self.grid[state].set_v(v[state], best_actions=actions, min_Q=min_value, max_Q=max_value)

    def update_q(self, Q, state=None, message=None):
        self.set_state(state)
        self.message.text = message
        m = [Q.max(s) for s in self.mdp.nonterminal_states]
        min_value, max_value = min(m), max(m)

        for state, grid in self.grid.items():
            tile = self.mdp.grid[state]
            if tile != '#':
                actions = self.mdp.A(state)
                g = self.grid[state]
                qs = [np.round(Q[state, a], 2) for a in actions]
                g.set_q(actions, qs, min_Q=min_value, max_Q=max_value)

    def set_state(self, state):
        i, j = state if isinstance(state, tuple) else (-5, -5)
        self.pacman.group.translate(self.MARGIN + self.GRID_SIZE * i, self.dy + self.GRID_SIZE * j)

    def draw(self):
        self.batch.draw()


def getColor(val, min_value, max_value):
    r = val * 0.65 / min_value if val < 0 and min_value < 0 else 0
    g = val * 0.65 / max_value if val > 0 and max_value > 0 else 0
    return int(r*255), int(g*255), 0


class BlankSquare(GroupedElement):
    BLANK = ' '
    EXIT = 'E'
    WALL = '#'
    action_labels, cross, center_text = None, None, None

    def __init__(self, type, grid_size, batch, pg=None, parent=None, order=0):
        self.type = type
        self.GRID_SIZE = grid_size
        self.elements = []
        super().__init__(batch, pg=pg, parent=parent, order=order)

    def set_q(self, actions, q_values, min_Q, max_Q):
        if self.type == 'E':
            self.set_v(q_values[0], best_actions=[], min_Q=min_Q, max_Q=max_Q)
        else:
            for a, q in zip(actions, q_values):
                self.action_labels[a]['label'].text = '%.2f' % q
                self.action_labels[a]['label'].color = WHITE+(255,) if q == max(q_values) else MUTED_TEXT_COLOR+(255,)
                self.action_labels[a]['triangle'].color = getColor(q, min_value=min_Q, max_value=max_Q)
                self.action_labels[a]['action'].visible = False
            self.center_text.text = ""
            for c in self.cross:
                c.visible = True

    def set_v(self, values, best_actions, min_Q, max_Q):
        for c in self.cross:
            c.visible = False
        q_value_str = '%.2f' % np.round(values, 2)
        color = getColor(values, min_value=min_Q, max_value=max_Q)
        self.center_text.text = q_value_str
        for a in self.action_labels:
            self.action_labels[a]['label'].text = ""
            self.action_labels[a]['triangle'].color = color
            self.action_labels[a]['action'].visible = a in best_actions

    def mkrect(self, side_width, line_width, bg):
        ps = [(x * side_width / 2, y * side_width / 2) for x, y in [(-1, 1), (1, 1), (1, -1), (-1, -1)]]
        for i in range(4):
            self.elements.append(
                Line(ps[i][0], ps[i][1], ps[(i + 3) % 4][0], ps[(i + 3) % 4][1], color=WHITE, width=line_width, **bg))
        return ps

    def render(self):
        s = (0, -0.5 * self.GRID_SIZE + 5)
        n = (0, 0.5 * self.GRID_SIZE - 5)
        w = (-0.5 * self.GRID_SIZE + 5, 0)
        e = (0.5 * self.GRID_SIZE - 5, 0)

        bg = dict(batch=self.batch, group=self.group)
        self.elements = []
        if self.type == '#':
            self.elements.append(Rectangle(-self.GRID_SIZE/2, -self.GRID_SIZE/2, self.GRID_SIZE, self.GRID_SIZE, color=OBSTACLE_COLOR, **bg))

        p_outline = self.mkrect(side_width=self.GRID_SIZE, line_width=3, bg=bg)
        bg2 = dict(batch=self.batch, group=CameraGroup(self.group.order+200, pg=self.group))

        if self.type == ' ' or self.type == 'E':
            self.action_labels = {}
            for k, ((x, y), anchor_x, anchor_y) in enumerate([(n, 'center', 'top'), (e, 'right', 'center'),
                                                             (s, 'center', 'baseline'), (w, 'left', 'center')]):
                self.action_labels[k] = {}

                triangle = Triangle(*p_outline[k], *p_outline[(k + 1) % 4], 0, 0, color=(0, 0, 0), **bg)

                l = pyglet.text.Label("0.00", x=x, y=y, font_name='Arial', font_size=int(20 * 0.75), bold=True,
                                      color=WHITE + (255,), anchor_x=anchor_x, anchor_y=anchor_y, **bg2)

                self.action_labels[k]['label'] = l
                self.action_labels[k]['triangle'] = triangle

                pts = [(x * self.GRID_SIZE, y * self.GRID_SIZE) for x, y in [(0, -0.45), (0.05, -0.40), (-0.05, -0.40)]]

                if k == 0 or k == 1:
                    pts = [(x, -y) for x, y in pts]

                if k == 1 or k == 3:
                    pts = [(y, x) for x, y in pts]

                self.action_labels[k]['action'] = Triangle(*pts[0], *pts[1], *pts[2], color=WHITE, **bg2)

            self.cross = [Line(p_outline[i][0], p_outline[i][1], p_outline[(i + 2) % 4][0], p_outline[(i + 2) % 4][1],
                               color=WHITE, width=2, **bg) for i in range(2)]

        if self.type == 'E':
            self.mkrect(side_width=self.GRID_SIZE*.8, line_width=2, bg=bg2)

        if self.type == 'E' or self.type == ' ':
            self.center_text = Label("0.00", x=0, y=0, font_name='Arial', font_size=int(30*0.75), bold=True,
                                     color=WHITE + (255,), anchor_x='center', anchor_y='center', **bg2)

# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
# graphicsUtils.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import numpy as np
import os
# import pyglet
# import irlc.utils.gym21.pyglet_rendering as rendering
# from irlc.utils.gym21.pyglet_rendering import Viewer
# from pyglet.gl import *
import pygame
from pygame import gfxdraw
import threading
import time
import pygame

ghost_shape = [
    (0, - 0.5),
    (0.25, - 0.75),
    (0.5, - 0.5),
    (0.75, - 0.75),
    (0.75, 0.5),
    (0.5, 0.75),
    (- 0.5, 0.75),
    (- 0.75, 0.5),
    (- 0.75, - 0.75),
    (- 0.5, - 0.5),
    (- 0.25, - 0.75)
]

def _adjust_coords(coord_list, x, y):
    for i in range(0, len(coord_list), 2):
        coord_list[i] = coord_list[i] + x
        coord_list[i + 1] = coord_list[i + 1] + y
    return coord_list

def formatColor(r, g, b):
    return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))

def colorToVector(color):
    return list(map(lambda x: int(x, 16) / 256.0, [color[1:3], color[3:5], color[5:7]]))

def h2rgb(color):
    if color is None or isinstance(color, tuple):
        return color
    if color.startswith("#"):
        color = color[1:]
    return tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4))

def h2rgb255(color):
    if isinstance(color, tuple):
        return color
    # c =
    return tuple(int(cc*255) for cc in h2rgb(color))
    if color is None:
        return None
    if color.startswith("#"):
        color = color[1:]
    return tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4))

class GraphicsCache:
    break_cache = False
    def __init__(self, viewer, verbose=False):
        self.viewer = viewer
        # self._items_in_viewer = {}
        # self._seen_things = set()
        self.clear()
        self.verbose = verbose

    def copy_all(self):
        self._seen_things.update( set( self._items_in_viewer.keys() ) )

    def clear(self):
        self._seen_things = set()
        self.viewer.geoms.clear()
        self._items_in_viewer = {}

    def prune_frame(self):
        s0 = len(self._items_in_viewer)
        self._items_in_viewer = {k: v for k, v in self._items_in_viewer.items() if k in self._seen_things }
        if self.verbose:
            print("removed", len(self._items_in_viewer) - s0,  "geom size", len(self._items_in_viewer))
        self.viewer.geoms = list( self._items_in_viewer.values() )
        self._seen_things = set()


    def add_geometry(self, name, geom):
        if self.break_cache:
            if self._items_in_viewer == None:
                self.viewer.geoms = []
                self._items_in_viewer = {}

        self._items_in_viewer[name] = geom
        self._seen_things.add(name)



class GraphicsUtilGym:
    viewer = None
    _canvas_xs = None      # Size of canvas object
    _canvas_ys = None
    _canvas_x = None      # Current position on canvas
    _canvas_y = None

    def begin_graphics(self, width=640, height=480, color=formatColor(0, 0, 0), title="02465 environment", local_xmin_xmax_ymin_ymax=None, verbose=False):
        """ Main interface for managing graphics.
            The local_xmin_xmax_ymin_ymax controls the (local) coordinate system which is mapped onto screen coordinates. I.e. specify this
            to work in a native x/y coordinate system. If not, it will default to screen coordinates familiar from Gridworld.
        """

        icon = os.path.dirname(__file__) + "/../utils/graphics/dtu_icon.png"
        pygame_icon = pygame.image.load(icon)
        pygame.display.set_icon(pygame_icon)
        screen_width = width
        screen_height = height
        pygame.init()
        pygame.display.init()

        self.screen = pygame.display.set_mode(
            (screen_width, screen_height)
        )
        self.screen_width = width
        self.screen_height = height

        pygame.display.set_caption(title)

        if height % 2 == 1:
            height += 1 # Must be divisible by 2.
        self._bg_color = color
        # viewer = Viewer(width=int(width), height=int(height))
        # viewer.window.set_caption(title)
        # self.viewer = viewer
        # self.gc = GraphicsCache(viewer, verbose=verbose)
        self._canvas_xs, self._canvas_ys = width - 1, height - 1
        self._canvas_x, self._canvas_y = 0, self._canvas_ys
        if local_xmin_xmax_ymin_ymax is None:
            # local_coordinates = []
            # This will align the coordinate system so it begins in the top-left corner.
            # This is the default behavior of pygame.
            local_xmin_xmax_ymin_ymax = (0, width, 0, height)
        self._local_xmin_xmax_ymin_ymax = local_xmin_xmax_ymin_ymax


        # self.demand_termination = False
        # self.demand_termination = threading.Event()
        # self.threading_opts = object() # persistence
        # self.threading_opts.time_since_last_update = 0
        self.demand_termination = threading.Event()
        self.pause_refresh = False

        self.ask_for_pause = False
        self.is_paused = False

        def refresh_window(gutils):
            refresh_interval_seconds = 0.1 # Miliseconds
            t0 = time.time()
            while not gutils.demand_termination.is_set():
                t1 = time.time()
                if t1 - t0 > refresh_interval_seconds:
                    if not self.ask_for_pause:
                        self.is_paused = False
                        pygame.display.update()
                    else:
                        self.is_paused = True
                    t0 = t1
                time.sleep(refresh_interval_seconds/100)

        # for _ in range(10):
        #     pygame.display.update()
        # time.sleep(0.04)
        self.refresh_thread = threading.Thread(target=refresh_window, args=(self, ))
        self.refresh_thread.start()
        # time.sleep(0.5)
        # self.demand_termination.set()
        # self.refresh_thread.join(timeout=1000)


    def close(self):
        self.demand_termination.set()
        self.refresh_thread.join(timeout=1000)
        pygame.display.quit()
        pygame.quit()
        # TH 2023: These two lines are super important.
        #  pdraw cache the fonts. So when pygame is loaded/quites,
        #  the font cache is not flushed. This is not a problem
        #  when determining the width of strings the font has seen,
        #  but causes a segfault with NEW strings.
        from irlc.utils import ptext
        ptext._font_cache = {}

        self.isopen = False

    def render(self):
        # Render the track, etc.
        print("Rendering method not overwritten.")
        pass

    def blit(self, render_mode=None):
        # if render_mode == 'rgb_array':
        #     self.demand_termination.set()
        #     self.refresh_thread.join(timeout=1000)

        # self.surf = pygame.transform.flip(self.surf, False, True)
        self.render()
        self.screen.blit(self.surf, (0, 0))

        if render_mode == "human":


            pygame.event.pump()
            pygame.display.flip()
        elif render_mode == "rgb_array":
            # self.render()
            # self.screen.blit(self.surf, (0, 0))


            # Perhaps pause thread?
            # sa2 = np.zeros( (100, 100, 3), dtype=np.dtype('uint8'))
            # sa2 = pygame.surfarray.pixels3d(self.screen)

            # ar = np.transpose(np.array(sa))
            # self.pause_refresh = True
            # self.ask_for_pause = True
            # while not self.is_paused:
            #     time.sleep(0.05)

            # self.demand_termination.set()
            # self.refresh_thread.join(timeout=1000)
            # time.sleep(0.05)
            # ar = np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

            # sa = np.zeros(sa2.shape, dtype=np.dtype('uint8'))
            # sa += sa2

            # ar = np.transpose(sa, axes=(1, 0, 2))
            # self.ask_for_pause = False
            # return ar
            # return np.transpose(ar),  axes=(1, 0, 2))
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def rectangle(self, color, x, y, width, height, border=0, fill_color=None):
        x2,y2 = self.fixxy((x+width, y+height))
        x, y = self.fixxy((x,y))

        c1 = min([x, x2])
        c2 = min([y, y2])

        w = abs(x-x2)
        h = abs(y - y2)

        pygame.draw.rect(self.surf, color, pygame.Rect( int(c1), int(c2), int(w), int(h)), border)


    def draw_background(self, background_color=None):
        if background_color is None:
            background_color = (0, 0, 0)
        self._bg_color = background_color

        # print("drawing bg")
        x1, x2, y1, y2 = self._local_xmin_xmax_ymin_ymax
        # corners = [(0,0), (0, self._canvas_ys), (self._canvas_xs, self._canvas_ys), (self._canvas_xs, 0)]
        corners = [ (x1, y1), (x2, y1), (x2, y2), (x1, y2)  ]
        # print(corners)
        # for s in corners:
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        # self.surf.fill((0, 0, 0))
        self.polygon(name="background", coords=corners, outlineColor=self._bg_color, fillColor=self._bg_color, filled=True, smoothed=False)

    def fixxy(self, xy):
        x,y = xy
        x = (x - self._local_xmin_xmax_ymin_ymax[0]) / (self._local_xmin_xmax_ymin_ymax[1] - self._local_xmin_xmax_ymin_ymax[0]) * self.screen.get_width()
        y = (y - self._local_xmin_xmax_ymin_ymax[2]) / (self._local_xmin_xmax_ymin_ymax[3] - self._local_xmin_xmax_ymin_ymax[2]) * self.screen.get_height()
        return int(x), int(y)


    def plot(self, name, x, y, color=None, width=1.0):
        coords = [(x_,y_) for (x_, y_) in zip(x,y)]
        if color is None:
            color = "#000000"
        return self.polygon(name, coords, outlineColor=color, filled=False, width=width)

    def polygon(self, name, coords, outlineColor=None, fillColor=None, filled=True, smoothed=1, behind=0, width=1.0, closed=False):
        c = []
        for coord in coords:
            c.append(coord[0])
            c.append(coord[1])

        coords = [self.fixxy(c) for c in coords]
        if fillColor == None: fillColor = outlineColor
        poly = None
        if not filled: fillColor = ""
        # from gym.envs.classic_control import PolyLine
        # from irlc.utils.gym21 import pyglet_rendering as rendering # import PolyLine, LineWidth, FilledPolygon, PolyLine
        c = [self.fixxy(tuple(c[i:i+2])) for i in range(0, len(c), 2)]
        if not filled:
            gfxdraw.polygon(self.surf, coords, h2rgb255(outlineColor))
            pygame.draw.polygon(self.surf, h2rgb255(outlineColor), coords, width=int(width))
            # poly = rendering.PolyLine(c, close=closed)
            # poly.set_linewidth(width)
            # poly.set_color(*h2rgb(outlineColor))
        else:
            gfxdraw.filled_polygon(self.surf, coords, h2rgb255(fillColor))
            # poly = rendering.FilledPolygon(c)
            # poly.set_color(*h2rgb(fillColor))
            # poly.add_attr(rendering.LineWidth(width))

        if outlineColor is not None and len(outlineColor) > 0 and filled: # Not sure why this cannot be merged with the filled case...
            # gfxdraw.polygon(self.surf, coords, h2rgb255(outlineColor), width=int(width))
            pygame.draw.polygon(self.surf, h2rgb255(outlineColor), coords, width=int(width))


            # outl = rendering.Poly(c, close=True)
            # outl.set_linewidth(width)
            # outl.set_color(*h2rgb(outlineColor))
        # if poly is not None:
        #     pass
        #     # self.gc.add_geometry(name, poly)
        # else:
        #     raise Exception("Bad polyline")
        return poly

    def square(self, name, pos, r, color, filled=1, behind=0):
        x, y = pos
        coords = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
        return self.polygon(name, coords, color, color, filled, 0, behind=behind)

    def centered_arc(self, color, pos, r, start_angle, stop_angle, width=1):
        # Draw a centered arc (pygame defaults to boxed arcs)
        # x, y = self.fixxy(pos)
        x, y = pos
        # pygame.Rect(x-r, y-r, 2*r, 2*r)

        # color, rect, start_angle, stop_angle, width = 1)
        # fuck this arc shit.
        tt = np.linspace(start_angle / 360 * 2 * np.pi,stop_angle / 360 * 2 * np.pi, int(r * 10))
        px = np.cos(tt) * r
        py = -np.sin(tt) * r
        pp = list(zip(px.tolist(), py.tolist()))
        # if style == 'pieslice':
        #     pp = [(0, 0), ] + pp + [(0, 0), ]
        pp = [((x + a, y + b)) for (a, b) in pp]
        # if style == 'arc':  # For pacman. I guess this one makes the rounded wall segments.
        pp = [self.fixxy(p_) for p_ in pp]

        pygame.draw.lines(self.surf, h2rgb255(color), False, pp, width)


        # pygame.draw.arc(self.surf, h2rgb255(color), pygame.Rect(x-r, y-r, 2*r, 2*r), start_angle/180*np.pi, stop_angle/180*np.pi, width)

        # self.ga.circle(name + "s4", add(screen2, (self.gridSize * 2 * WALL_RADIUS, self.gridSize * (-2) * WALL_RADIUS)),
        #                WALL_RADIUS * self.gridSize - 1, wallColor, wallColor, (180, 271), 'arc')

        pass

    # def rotate_around(self, x, y, xy0, angle_degrees=0):
    #     return x, y

    def circle(self, name, pos, r, outlineColor=None, fillColor=None, endpoints=None, style='pieslice', width=2):
        pos = self.fixxy(pos)
        x, y = pos
        if endpoints == None:
            e = [0, 359]
        else:
            e = list(endpoints)
        while e[0] > e[1]: e[1] = e[1] + 360
        if endpoints is not None and len(endpoints) > 0:
            tt = np.linspace(e[0]/360 * 2*np.pi, e[-1]/360 * 2*np.pi, int(r*20) )
            px = np.cos(tt) * r
            py = -np.sin(tt) * r
            pp = list(zip(px.tolist(), py.tolist()))
            if style == 'pieslice':
                pp = [(0,0),] + pp + [(0,0),]
            pp = [( (x+a, y+b)) for (a,b) in pp  ]
            if style == 'arc': # For pacman. I guess this one makes the rounded wall segments.
                pp = [self.fixxy(p_) for p_ in pp]
                pygame.draw.lines(self.surf, outlineColor, False, pp, width)
                # self.polygon(name, pp, fillColor=None, filled=False, outlineColor=outlineColor, width=width)
                # outl = rendering.PolyLine(pp, close=False)
                # outl.set_linewidth(width)
                # outl.set_color(*h2rgb(outlineColor))
                # self.gc.add_geometry(name, outl)
            elif style == 'pieslice':
                # h2rgb()
                # gfxdraw.filled_circle(self.surf, int(pos[0]), int(pos[1]), int(r), h2rgb255(fillColor))

                self.polygon(name, pp, fillColor=fillColor, outlineColor=outlineColor, width=width)
            else:
                raise Exception("bad style", style)
        else:
            # xy = self.fixxy(pos)
            # x, y = pos
            # pygame.draw.

            gfxdraw.filled_circle(self.surf, x, y, int(r), h2rgb255(fillColor))
            # gfxdraw.aacircle()


            # gfxdraw.filled_circle()
            # circ = rendering.make_circle(r)
            # circ.set_color(*h2rgb(fillColor))
            # tf = rendering.Transform(translation = self.fixxy(pos))
            # circ.add_attr(tf)
            # self.gc.add_geometry(name, circ)


    # def moveCircle(self, id, pos, r, endpoints=None):
    #     # global _canvas_x, _canvas_y
    #     x, y = pos
    #     x0, x1 = x - r - 1, x + r
    #     y0, y1 = y - r - 1, y + r
    #     if endpoints == None:
    #         e = [0, 359]
    #     else:
    #         e = list(endpoints)
    #     while e[0] > e[1]: e[1] = e[1] + 360
    #     self.edit(id, ('start', e[0]), ('extent', e[1] - e[0]))
    #
    # def edit(id, *args):
    #     pass


    def text(self, name, pos, color, contents, font='Helvetica', size=12, style='normal', anchor="w"):
        pos = self.fixxy(pos)
        ax = "center"
        ax = "left" if anchor == "w" else ax
        # ax = "right" if anchor == "e" else ax
        ay = "center"
        ay = "baseline" if anchor == "s" else ay
        # ay = "top" if anchor == "n" else ay
        # psz = int(-size * 0.75) if size < 0 else size
        # cl = tuple(int(c*255) for c in h2rgb(color) )+(255,)
        # print("Missing")

        # fonts = pygame.font.get_fonts()
        # print(len(fonts))
        from irlc.utils.ptext import draw
        # anchor = ("center", "left")
        # opts = {}
        if anchor == 'w':
            opts = dict(midleft=pos)
        elif anchor == 'e':
            opts = dict(midright=pos)
        elif anchor == 's':
            opts = dict(midbottom=pos)
        elif anchor == 'n':
            opts = dict(midtop=pos)
        elif anchor == 'c':
            opts = dict(center=pos)
        else:
            raise Exception("Unknown anchor", anchor)

        draw(contents, surf=self.surf, color=h2rgb255(color), pos=pos, **opts)

        # font = pygame.font.Font(None, psz)
        # t = font.render(contents, False, h2rgb255(color))
        # a = 324

        # self.surf.blit(t, pos)

        # screen.blit(name, (40, 140))
        # screen.blit(game_over, (40, 240))

        # for f in fonts:
        #     if "times" in f or "verdena" in f or "arial" in f:
        #
        #         print(f)
        return
        # label = pyglet.text.Label(contents, x=int(x_), y=int(y_),  font_name='Arial', font_size=psz, bold=style=="bold",
        #                           color=cl,
        #                           anchor_x=ax, anchor_y=ay)
        # self.gc.add_geometry(name, TextGeom(label))

    def line(self, name, here, there, color=formatColor(0, 0, 0), width=2):
        # print(color)
        here, there = self.fixxy(here), self.fixxy(there)
        pygame.draw.line(self.surf, h2rgb255(color), here, there, width)
        # pygame.draw.line(self.surf, h2rgb255(color), self.fixxy(here), self.fixxy(there), width)
        # return
        # poly = MyLine(self.fixxy(here), self.fixxy(there), width=width)
        # poly.set_color(*h2rgb(color))
        # poly.add_attr(rendering.LineWidth(width))
        # self.gc.add_geometry(name, poly)
        # return None

# class MyLine(object): #rendering.Geom):
#     def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0), width=1):
#         rendering.Geom.__init__(self)
#         self.start = start
#         self.end = end
#         self.linewidth = rendering.LineWidth(width)
#         self.add_attr(self.linewidth)
#
#     def render1(self):
#         glBegin(GL_LINES)
#         glVertex2f(*self.start)
#         glVertex2f(*self.end)
#         glEnd()
#
# class TextGeom(object): # rendering.Geom):
#     def __init__(self, label):
#         super().__init__()
#         self.label = label
#
#     def render(self, batch=None):
#         self.label.draw()

# 547 lines
# 556 lines
# 270


# IMAGE = pygame.image.load('an_image.png').convert_alpha()
# class Player(pygame.sprite.Sprite):
#     def __init__(self, pos):
#         super().__init__()
#         self.image = IMAGE
#         self.rect = self.image.get_rect(center=pos)


def rotate_around(pos, xy0, angle):
    if isinstance(pos, list) and isinstance(pos[0], tuple):
        return [rotate_around(p, xy0, angle) for p in pos]
    return ((pos[0] - xy0[0]) * np.cos(angle / 180 * np.pi) - (pos[1] - xy0[1]) * np.sin(angle / 180 * np.pi) + xy0[0],
            (pos[0] - xy0[0]) * np.sin(angle / 180 * np.pi) + (pos[1] - xy0[1]) * np.cos(angle / 180 * np.pi) + xy0[1])

# Class for creating test object

class Object(pygame.sprite.Sprite):
    def __init__(self, file, image_width=None, graphics=None):
        super(Object, self).__init__()
        fpath = os.path.dirname(__file__) +"/graphics/"+file
        image = pygame.image.load(fpath).convert_alpha()
        if image_width is not None:
            image_height = int( image_width / image.get_width() * image.get_height() )
            self.og_surf = pygame.transform.smoothscale(image, (image_width, image_height))
            # raise Exception("Implement this")
        else:
            self.og_surf = image
        # self.og_surf = pygame.transform.smoothscale(image, (100, 100))
        self.surf = self.og_surf
        self.rect = self.surf.get_rect(center=(400, 400))
        self.ga = graphics

    def move_center_to_xy(self, x, y):
        # Note: These are in the local coordinate system coordinates.
        x,y = self.ga.fixxy((x,y))
        self.rect.center = (x,y)

    # THE MAIN ROTATE FUNCTION
    def rotate(self, angle):
        """ Rotate sprite around it's center. """
        self.angle = angle
        self.surf = pygame.transform.rotate(self.og_surf, self.angle)
        # self.angle += self.change_angle
        # self.angle = self.angle % 360
        self.rect = self.surf.get_rect(center=self.rect.center)

    def blit(self, surf):
        surf.blit(self.surf, self.rect)


class UpgradedGraphicsUtil(GraphicsUtilGym):
    def __init__(self, screen_width=800, screen_height=None, xmin=0., xmax=800., ymin=0., ymax=600., title="Gym window"):
        if screen_height is None:
            screen_height = int(screen_width / (xmax - xmin) * (ymax-ymin))
        elif xmin is None:
            xmin = 0
            xmax = screen_width
            ymin = 0
            ymax = screen_height
        else:
            raise Exception()
        self.begin_graphics(width=screen_width, height=screen_height, local_xmin_xmax_ymin_ymax=(xmin, xmax, ymin, ymax), title=title)

    def get_sprite(self, name):
        """ Load a sprite from the graphics directory. """

        pass

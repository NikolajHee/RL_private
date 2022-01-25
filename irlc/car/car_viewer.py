# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from pyglet.shapes import Rectangle, Circle
from irlc.utils.pyglet_rendering import PygletViewer, PolygonOutline, GroupedElement


class CarViewer(PygletViewer):
    def __init__(self, car):
        track_outline = (0, 0, 0)
        track_middle = (220, 25, 25)

        n = int(10 * (car.map.PointAndTangent[-1, 3] + car.map.PointAndTangent[-1, 4]))
        center = [car.map.getGlobalPosition(i * 0.1, 0) for i in range(n)]
        outer = [car.map.getGlobalPosition(i * 0.1, -car.map.width) for i in range(n)]
        inner = [car.map.getGlobalPosition(i * 0.1, car.map.width) for i in range(n)]

        fudge = 0.2
        xs, ys = zip(*outer)
        super().__init__(screen_width=1000, xmin=min(xs) - fudge, xmax=max(xs) + fudge,
                         ymin=min(ys) - fudge, ymax=max(ys) + fudge, title="Car on a racetrack")

        batch = self.batch
        self.track = [PolygonOutline(batch, coords=inner, outlineColor=track_outline, width=10),
                      PolygonOutline(batch, coords=outer, outlineColor=track_outline, width=2),
                      PolygonOutline(batch, coords=center, outlineColor=track_middle, width=2),
                      PolygonOutline(batch, coords=[inner[0], outer[0]], outlineColor=track_outline, width=1)]
        self.car = CarGeom(batch, order=1)

    def update(self, xglob):
        x, y, psi = xglob[4], xglob[5], xglob[3]
        self.car.group.translate(x, y)
        self.car.group.rotate(psi)


class CarGeom(GroupedElement):
    def render(self):
        # BorderedRectangle does not work for some reason..
        width = 0.4*2
        height = 0.2*2
        dd = width/10
        self.cm1 = Rectangle(-(width+dd)/2, -(height+dd)/2, width+dd, height+dd, color=(0, 0, 0), batch=self.batch, group=self.group)
        self.cm2 = Rectangle(-width/2, -height/2, width, height, color=(220, 50, 50), batch=self.batch, group=self.group)
        self.cm3 = Circle(0, 0, width/10, color=(0, 0, 0), batch=self.batch, group=self.group)

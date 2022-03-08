# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
""" This file contains code you can either use (or not) to render the R2D2 robot.
If you want to use it, the recommended way is to build a class similar to ... and let it contain code such as:


from irlc.project2.utils import render_car

class R2D2RobotWithRendering():
    def __init__(...)
        ...
        ...

    ...

    def render(self, x, mode="human"): 
        return render_car(self, x, x_target=self.x_target, mode=mode) 

    ...

This will add visualization functionality. To use it, do:

from irlc import VideoMonitor
env = VideoMonitor(R2D2RobotWithRendering())

"""


def render_car(env, x, x_target, mode="human"):
    screen_width = 600
    screen_height = 400
    x_lim = [-10, 10]
    # if ylim is None:
    #     ylim = [-10,10]
    world_width = x_lim[1] - x_lim[0]
    scale = screen_width / world_width
    polewidth = 10.0
    cartwidth = 50.0 * 0.6
    cartheight = 30.0 * 0.6
    def p2glob(p):
        return ( p[0]*scale + screen_width/2, p[1]*scale + screen_height/2 )

    def mkX(rendering, x_target, dd=0.5, color=(0,0,0) ):
        l = [rendering.Line(p2glob((x_target[0] - dd, x_target[1])), p2glob((x_target[0] + dd, x_target[1]))),
             rendering.Line(p2glob((x_target[0], x_target[1] - dd)), p2glob((x_target[0], x_target[1] + dd)))]
        for dl in l:
            dl.set_color(*color)
        return l
    if not hasattr(env, 'viewer'):
        env.viewer = None

    if env.viewer is None:
        from gym.envs.classic_control import rendering
        env.viewer = rendering.Viewer(screen_width, screen_height)
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        env.carttrans = rendering.Transform()
        cart.add_attr(env.carttrans)
        env.viewer.add_geom(cart)
        env.poletrans = rendering.Transform(translation=(0, axleoffset))
        env.axle = rendering.make_circle(polewidth / 2)
        env.axle.add_attr(env.carttrans)
        env.axle.set_color(.5, .5, .8)
        env.viewer.add_geom(env.axle)
        for dl in mkX(rendering, (0,0), dd=10, color=(0,0,0)):
            env.viewer.add_geom(dl)
        if x_target is not None:
            for dl in mkX(rendering, x_target, dd=0.5, color=(1,0,0)):
                env.viewer.add_geom(dl)

    cartx = x[0] * scale + screen_width / 2.0
    carty = x[1] * scale + screen_height / 2.0
    env.carttrans.set_translation(cartx, carty)
    env.carttrans.set_rotation(x[2])
    return env.viewer.render(return_rgb_array=mode == 'rgb_array')

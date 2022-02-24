# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.car.car_viewer import CarViewer
import numpy as np
import sympy as sym
from gym.spaces.box import Box
from scipy.optimize import Bounds
from irlc.car.sym_map import SymMap, wrap_angle
from irlc.ex04.continuous_time_model import ContiniousTimeSymbolicModel
from irlc.ex04.cost_continuous import SymbolicQRCost
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment

class SymbolicBicycleModel(ContiniousTimeSymbolicModel):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    def __init__(self, map_width=0.8, simple_bounds=None, cost=None, hot_start=False, verbose=True):
        s = """
        Coordinate system of the car:
        State x consist of
        x[0] = Vx (speed in direction of the car body)
        x[1] = Vy (speed perpendicular to car body)
        x[2] = wz (Yaw rate; how fast the car is turning)
        x[3] = e_psi (Angle of rotation between car body and centerline)
        x[4] = s (How far we are along the track)
        x[5] = e_y (Distance between car body and closest point on centerline)

        Meanwhile the actions are
        u[0] : Angle between wheels and car body (i.e. are we steering to the right or to the left)
        u[1] : Engine force (applied to the rear wheels, i.e. accelerates car)
        """
        if verbose:
            print(s)
        if simple_bounds is None:
            simple_bounds = dict()
        self.map = SymMap(width=map_width)
        v_max = 3.0

        self.viewer = None  # rendering
        self.hot_start = hot_start
        self.observation_space = Box(low=np.asarray([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -map_width], dtype=float),
                                     high=np.asarray([v_max, np.inf, np.inf, np.inf, np.inf, map_width]), dtype=float)
        self.action_space = Box(low=np.asarray([-0.5, -1]), high=np.asarray([0.5, 1]), dtype=float)

        xl = np.zeros((6,))
        xl[4] = self.map.TrackLength
        simple_bounds = {'x0': Bounds(list(self.reset()), list(self.reset())),
                        'xF': Bounds(list(xl), list(xl)), **simple_bounds}

        if cost is None:
            cost = SymbolicQRCost(Q=np.zeros((self.state_size,self.state_size)), R=np.eye(self.action_size), qc=1.)

        super().__init__(cost=cost, simple_bounds=simple_bounds)

    def render(self, x, mode="human"):
        if self.viewer == None:
            self.viewer = CarViewer(self)

        self.viewer.update(self.x_curv2x_XY(x))
        # print(x)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def reset(self):
        x0 = np.zeros((6,))
        if self.hot_start:
            x0[0] = 0.5  # Start velocity is 0.5
        return x0

    def x_curv2x_XY(self, x_curv):
        '''
        Utility function for converting x (including velocities, etc.) from local (curvilinear) coordinates to global XY position.
        '''
        Xc, Yc, vangle = self.map.getGlobalPosition(s=x_curv[4], ey=x_curv[5], epsi=x_curv[3])
        dglob = np.asarray([x_curv[0], x_curv[1], x_curv[2], vangle, Xc, Yc])
        return dglob

    def sym_f(self, x, u, t=None, curvelinear_coordinates=True, curvature_s=None):
        '''
        Create derivative function

        \dot{x} = f(x, u)

        We will both create it in curvelinear coordinates or normal (global) coordinates.
        '''
        # Vehicle Parameters
        m = 1.98
        lf = 0.125
        lr = 0.125
        Iz = 0.024
        Df = 0.8 * m * 9.81 / 2.0
        Cf = 1.25
        Bf = 1.0
        Dr = 0.8 * m * 9.81 / 2.0
        Cr = 1.25
        Br = 1.0

        vx = x[0]
        vy = x[1]
        wz = x[2]
        if curvelinear_coordinates:
            epsi = x[3]
            s = x[4]
            ey = x[5]
        else:
            psi = x[3]

        delta = u[0]
        a = u[1]

        alpha_f = delta - sym.atan2(vy + lf * wz, vx)
        alpha_r = -sym.atan2(vy - lf * wz, vx)

        # Compute lateral force at front and rear tire
        Fyf = 2 * Df * sym.sin(Cf * sym.atan(Bf * alpha_f))
        Fyr = 2 * Dr * sym.sin(Cr * sym.atan(Br * alpha_r))

        d_vx = (a - 1 / m * Fyf * sym.sin(delta) + wz * vy)
        d_vy = (1 / m * (Fyf * sym.cos(delta) + Fyr) - wz * vx)
        d_wz = (1 / Iz * (lf * Fyf * sym.cos(delta) - lr * Fyr))

        if curvelinear_coordinates:
            cur = self.map.sym_curvature(s)
            d_epsi = (wz - (vx * sym.cos(epsi) - vy * sym.sin(epsi)) / (1 - cur * ey) * cur)
            d_s = ((vx * sym.cos(epsi) - vy * sym.sin(epsi)) / (1 - cur * ey))
            """
            Compute derivative of e_y here (d_ey). See paper for details. 
            """
            d_ey = (vx * sym.sin(epsi) + vy * sym.cos(epsi)) # Old ex here ! b ! b
            # implement the ODE governing ey (distane from center of road) in curveliner coordinates
            xp = [d_vx, d_vy, d_wz, d_epsi, d_s, d_ey]

        else:
            d_psi = wz
            d_X = ((vx * sym.cos(psi) - vy * sym.sin(psi)))
            d_Y = (vx * sym.sin(psi) + vy * sym.cos(psi))

            xp = [d_vx, d_vy, d_wz, d_psi, d_X, d_Y]
        return xp

    def fix_angles(self, x):
        # fix angular component of x
        if x.size == self.state_size:
            x[3] = wrap_angle(x[3])
        elif x.shape[1] == self.state_size:
            x[:,3] = wrap_angle(x[:,3])
        return x


class DiscreteCarModel(DiscretizedModel): 
    def __init__(self, dt=0.1, cost=None, **kwargs): 
        model = SymbolicBicycleModel(**kwargs)
        self.observation_space = model.observation_space
        self.action_space = model.action_space 
        if cost is None:
            from irlc.ex04.cost_discrete import DiscreteQRCost
            cost = DiscreteQRCost(env=self, Q=np.zeros((self.state_size, self.state_size)), R=np.eye(self.action_size))
        super().__init__(model=model, dt=dt, cost=cost)

        self.cost = cost
        self.map = model.map


class CarEnvironment(ContiniousTimeEnvironment): 
    def __init__(self, Tmax=10, noise_scale=1.0, cost=None, max_laps=10, hot_start=False, **kwargs):
        discrete_model = DiscreteCarModel(cost=cost, hot_start=hot_start, **kwargs)
        super().__init__(discrete_model, Tmax=Tmax) 
        self.map = discrete_model.map
        self.noise_scale = noise_scale
        self.cost = cost
        self.completed_laps = 0
        self.max_laps = max_laps

    def simple_bounds(self):
        simple_bounds = {'x': Bounds(self.observation_space.low, self.observation_space.high),
                         't0': Bounds([0], [0]),
                         'u': Bounds(self.action_space.low, self.action_space.high)}
        return simple_bounds

    """ Backwards compatibility bullshit """
    def step(self, u):
        # return self.old_step(x=self.state, u=u, seed=self.seed)
        xp, cost, done, meta = super().step(u)
        x = xp
        if hasattr(self, 'seed') and self.seed is not None and not callable(self.seed):
            np.random.seed(self.seed)

        noise_vx = np.maximum(-0.05, np.minimum(np.random.randn() * 0.01, 0.05))
        noise_vy = np.maximum(-0.1, np.minimum(np.random.randn() * 0.01, 0.1))
        noise_wz = np.maximum(-0.05, np.minimum(np.random.randn() * 0.005, 0.05))
        if True: #self.noise_scale > 0:
            x[0] = x[0] + 0.03 * noise_vx #* self.noise_scale
            x[1] = x[1] + 0.03 * noise_vy #* self.noise_scale
            x[2] = x[2] + 0.03 * noise_wz #* self.noise_scale

        if x[4] > self.map.TrackLength:
            self.completed_laps += 1
            x[4] -= self.map.TrackLength

        done = self.completed_laps >= self.max_laps
        if x[4] < 0:
            assert(False)
        return x, cost, done, meta

    def L(self, x):
        '''
        Implement whether we have obtained the terminal condition. see eq. 4 in "Autonomous Racing using LMPC"

        :param x:
        :return:
        '''
        return x[4] > self.map.TrackLength

    def epoch_reset(self, x):
        '''
        After completing one epoch, i.e. when L(x) == True, reset the x-vector using this method to
        restart the epoch. In practice, take one more lap on the track.

        :param x:
        :return:
        '''
        x = x.copy()
        x[4] -= self.map.TrackLength
        return x

if __name__ == "__main__":
    # car = SymbolicBicycleModel()
    # car.render(car.reset())
    # sleep(2.0)
    # car.close()
    # print("Hello world")
    env = CarEnvironment()
    env.metadata['video.frames_per_second'] = 10000
    from irlc import VideoMonitor
    # env = wrappers.Monitor(env, "carvid2", force=True, video_callable=lambda episode_id: True)
    env = VideoMonitor(env)
    env.reset()
    import time
    t0 = time.time()
    n = 100
    for _ in range(n):
        u = env.action_space.sample()
        # print(u)
        # u *= 0
        u[0] = 0
        u[1] = 0.01
        s, cost, done, meta = env.step(u)
        # print(s)
        # sleep(5)
    env.close()
    tpf = (time.time()- t0)/n
    print("TPF", tpf, "fps", 1/tpf)

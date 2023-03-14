# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import sympy as sym
import numpy as np
import gym
from gym.spaces import Box
from irlc.ex04.continuous_time_model import ContiniousTimeSymbolicModel
from irlc.ex04.cost_continuous import SymbolicQRCost
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment

class ContiniousCartpole(ContiniousTimeSymbolicModel):
    state_size = 4
    action_size = 1
    state_labels = ["$x$", r"$\frac{dx}{dt}$", r"$\theta$", r"$\frac{d \theta}{dt}$"]
    action_labels = ["Cart force $u$"]

    def __init__(self, mc=2,
                 mp=0.5,
                 l=0.5,
                 g=9.81, maxForce=50, dist=1.0, cost=None, bounds=None):

        self.mc = mc
        self.mp = mp
        self.l = l
        self.g = g
        self.maxForce = maxForce
        '''
        Default to kellys swingup task. (From matlab repo)
        '''
        # c0, _, _, _ = kelly_swingup(maxForce=maxForce, dist=dist)
        c0 = SymbolicQRCost(R=np.eye(1) * 0, Q=np.eye(4) * 0, qc=1)  # just minimum time

        # bounds_ = dict(tF_low=0.01, tF_high=np.inf,
        #               x_low=[-2 * dist, -np.inf, -2 * np.pi, -np.inf], x_high=[2 * dist, np.inf, 2 * np.pi, np.inf],
        #               u_low=[-maxForce], u_high=[maxForce],
        #               x0_low=[0, 0, np.pi, 0], x0_high=[0, 0, np.pi, 0],
        #               xF_low=[dist, 0, 0, 0], xF_high=[dist, 0, 0, 0])
        bounds_ = dict(tF_low=0, tF_high=np.inf,
                       x_low=-3, x_high=3,
                       u_low=-maxForce, u_high=maxForce)
        if cost is None:
            cost = c0
        if bounds is None:
            bounds = {}
        bounds = {**bounds_, **bounds}

        super().__init__(cost=cost, bounds=bounds)
        self.cp_render = None


    def close(self):
        if self.cp_render is not None:
            self.cp_render.close()

    def render(self, x, render_mode="human"):
        if self.cp_render is None:
            self.cp_render = gym.make("CartPole-v1", render_mode=render_mode)  # environment only used for rendering. Change to v1 in gym 0.26.
            self.cp_render.max_time_limit = 10000
            self.cp_render.reset()
        self.cp_render.unwrapped.state = np.asarray(x)  # environment is wrapped
        return self.cp_render.render()


    def sym_f(self, x, u, t=None):
        mp = self.mp
        l = self.l
        mc = self.mc
        g = self.g

        x_dot = x[1]
        theta = x[2]
        sin_theta = sym.sin(theta)
        cos_theta = sym.cos(theta)
        theta_dot = x[3]
        F = u[0]
        # Define dynamics model as per Razvan V. Florian's
        # "Correct equations for the dynamics of the cart-pole system".
        # Friction is neglected.

        # Eq. (23)
        temp = (F + mp * l * theta_dot ** 2 * sin_theta) / (mc + mp)
        numerator = g * sin_theta - cos_theta * temp
        denominator = l * (4.0 / 3.0 - mp * cos_theta ** 2 / (mc + mp))
        theta_dot_dot = numerator / denominator

        # Eq. (24)
        x_dot_dot = temp - mp * l * theta_dot_dot * cos_theta / (mc + mp)
        xp = [x_dot,
              x_dot_dot,
              theta_dot,
              theta_dot_dot]
        return xp

    def guess(self):
        guess = {'t0': 0,
                 'tF': 2,
                 'x': [np.asarray( self.bounds['x0_low'] ), np.asarray( self.bounds['xF_high'] )  ],
                 'u': [ np.asarray( [0] ), np.asarray(  [0] ) ] }
        return guess


    def kelly_swingup(maxForce=50, dist=1.0):
        """
        Return problem roughly comparable to the Kelly swingup task
        note we have to flip coordinate system because we are using corrected dynamics.
        https://github.com/MatthewPeterKelly/OptimTraj/blob/master/demo/cartPole/MAIN_minTime.m

        Use the SymbolicQRCost to get the cost function.
        """
        # TODO: 6 lines missing.
        bounds = dict(tF_low=0.01, tF_high=np.inf,
                    x_low=[-2 * dist, -np.inf, -2 * np.pi, -np.inf], x_high=[2 * dist, np.inf, 2 * np.pi, np.inf],
                    u_low=[-maxForce], u_high=[maxForce],
                    x0_low=[0, 0, np.pi, 0], x0_high=[0, 0, np.pi, 0],
                    xF_low=[dist, 0, 0, 0], xF_high=[dist, 0, 0, 0])  

        cost = SymbolicQRCost(R=np.eye(1)*0, Q=np.eye(4)*0, qc=1)  # just minimum time
        args = {}
        return cost, None, args, bounds




#raise NotImplementedError("Insert your solution and remove this error.")



def _cartpole_discrete_cost(model):
    from irlc.ex04.cost_discrete import DiscreteQRCost
    pole_length = model.continuous_model.l

    state_size = model.state_size
    Q = np.eye(state_size)
    Q[0, 0] = 1.0
    Q[1, 1] = Q[4, 4] = 0.
    Q[0, 2] = Q[2, 0] = pole_length
    Q[2, 2] = Q[3, 3] = pole_length ** 2

    print("Warning: I altered the cost-matrix to prevent underflow. This is not great.")
    R = np.array([[0.1]])
    Q_terminal = 1 * Q

    q = np.asarray([0,0,0,-1,0])
    # Instantaneous control cost.
    c3 = DiscreteQRCost(Q=Q*0, R=R * 0.1, q=1 * q, qN=q * 1)
    c3 += c3.goal_seeking_cost(Q=Q, x_target=model.x_upright)
    c3 += c3.goal_seeking_terminal_cost(QN=Q_terminal, xN_target=model.x_upright)
    cost = c3
    return cost

class GymSinCosCartpoleModel(DiscretizedModel): 
    state_labels =  ['x', 'd_x', '$\sin(\\theta)$', '$\cos(\\theta)$', '$d\\theta/dt$']
    action_labels = ['Torque $u$']

    def __init__(self, dt=0.02, cost=None, transform_actions=True, **kwargs): 
        model = ContiniousCartpole(**kwargs)
        self.transform_actions = transform_actions
        super().__init__(model=model, dt=dt, cost=cost) 
        self.x_upright = np.asarray(self.continious_states2discrete_states(model.bounds['xF_low'] ))
        if cost is None:
            cost = _cartpole_discrete_cost(self)
        self.cost = cost

    @property
    def max_force(self):
        return self.continuous_model.maxForce

    def sym_discrete_xu2continious_xu(self, x, u):
        x, dx, sin_theta, cos_theta, theta_dot = x[0], x[1], x[2], x[3], x[4]
        torque = sym.tanh(u[0]) * self.max_force if self.transform_actions else u[0]
        theta = sym.atan2(sin_theta, cos_theta)  # Obtain angle theta from sin(theta),cos(theta)
        return [x, dx, theta, theta_dot], [torque]

    def sym_continious_xu2discrete_xu(self, x, u):
        x, dx, theta, theta_dot = x[0], x[1], x[2], x[3]
        torque = sym.atanh(u[0]/self.max_force) if self.transform_actions else u[0]
        return [x, dx, sym.sin(theta), sym.cos(theta), theta_dot], [torque] 


class GymSinCosCartpoleEnvironment(ContiniousTimeEnvironment): 
    def __init__(self, Tmax=5, transform_actions=True, supersample_trajectory=False, render_mode='human', **kwargs):
        discrete_model = GymSinCosCartpoleModel(transform_actions=transform_actions, **kwargs)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(5,), dtype=float)
        if transform_actions:
            self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=float)
        else:
            self.action_space = Box(low=discrete_model.continuous_model.bounds['u_low'][0],
                                    high=discrete_model.continuous_model.bounds['u_low'][0],
                                    shape=(1,),dtype=float)
        super().__init__(discrete_model, Tmax=Tmax,render_mode=render_mode, supersample_trajectory=supersample_trajectory) 
        a = 234

    def step(self, u):
        self.discrete_model.continuous_model.u_prev = u
        return super().step(u)


class GymThetaCartpoleModel(DiscretizedModel):
    state_labels =  ['x', 'd_x', '$\theta$', '$d\theta/dt$']
    action_labels = ['Torque $u$']

    def __init__(self, dt=0.02, cost=None, **kwargs):
        model = ContiniousCartpole(**kwargs)
        super().__init__(model=model, dt=dt, cost=cost)
        self.x_upright = np.asarray(model.bounds['xF_low'] )

    @property
    def max_force(self):
        return self.continuous_model.maxForce


class GymThetaCartpoleEnvironment(ContiniousTimeEnvironment):
    def __init__(self, Tmax=5, transform_actions=True, supersample_trajectory=False, render_mode='human', **kwargs):
        discrete_model = GymThetaCartpoleModel(transform_actions=transform_actions, **kwargs)
        super().__init__(discrete_model, Tmax=Tmax, supersample_trajectory=supersample_trajectory, render_mode=render_mode)

    @property
    def max_force(self):
        return self.discrete_model.max_force

    def step(self, u):
        self.discrete_model.continuous_model.u_prev = u
        return super().step(u)

if __name__ == "__main__":
    from irlc import train, VideoMonitor
    from irlc import Agent
    env = GymSinCosCartpoleEnvironment()
    agent = Agent(env)
    env = VideoMonitor(env)
    stats, traj = train(env, agent, num_episodes=1, max_steps=100)
    env.close()

import sympy as sym
from irlc.ex04.continuous_time_model import ContiniousTimeSymbolicModel
from irlc.ex04.cost_continuous import SymbolicQRCost
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
import gym
from gym.spaces.box import Box
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment
import numpy as np

"""
SEE: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
"""
class ContiniousPendulumModel(ContiniousTimeSymbolicModel): #!s=a #!s=a
    state_labels= [r"$\theta$", r"$\frac{d \theta}{dt}$"]
    action_labels = ['Torque $u$']
    x_upright, x_down = np.asarray([0.0, 0.0]), np.asarray([np.pi, 0.0])

    def __init__(self, l=1., m=.8, g=9.82, friction=0.0, max_torque=6.0, cost=None, bounds=None): #!s=a
        self.g, self.l, self.m, self.max_torque = g, l, m, max_torque
        if bounds is None:
            bounds = {'tF_low': 0.5,                 'tF_high': 4,  #!s=bounds
                     't0_low': 0,                   't0_high': 0,
                     'x_low': [-2 * np.pi, -np.inf],'x_high': [2 * np.pi, np.inf],
                     'u_low': [-max_torque],        'u_high': [max_torque],
                     'x0_low': [np.pi, 0],          'x0_high': [np.pi, 0],
                     'xF_low': [0, 0],              'xF_high': [0, 0] } #!s=bounds
        if cost is None:
            cost = SymbolicQRCost(R=np.ones( (1,1) ), Q=np.eye(2) )
        super().__init__(cost=cost, bounds=bounds) #!s=a
        self.u_prev = None                        # For rendering
        self.cp_render = None
        self.friction = friction

    def render(self, x, render_mode="human"):
        if self.cp_render is None:
            self.cp_render = gym.make("Pendulum-v1", render_mode=render_mode)  # environment only used for rendering
            self.cp_render.max_time_limit = 10000
            self.cp_render.reset()

        self.cp_render.unwrapped.last_u = float(self.u_prev) if self.u_prev is not None else self.u_prev
        self.cp_render.unwrapped.state = np.asarray(x)
        return self.cp_render.render()

    def sym_f(self, x, u, t=None): #!s=a
        g, l, m = self.g, self.l, self.m
        theta_dot = x[1]  # Parameterization: x = [theta, theta']
        theta_dot_dot =  g/l * sym.sin(x[0]) + 1/(m*l**2) * u[0]
        return [theta_dot, theta_dot_dot] #!s=a

    def close(self):
        if self.cp_render is not None:
            self.cp_render.close()

guess = {'t0': 0,
         'tF': 2.5,
         'x': [np.asarray([0, 0]), np.asarray([np.pi, 0])],
         'u': [np.asarray([0]), np.asarray([0])] }

def _pendulum_cost(model):
    from irlc.ex04.cost_discrete import DiscreteQRCost
    Q = np.eye(model.state_size)
    Q[0, 1] = Q[1, 0] = model.l
    Q[0, 0] = Q[1, 1] = model.l ** 2
    Q[2, 2] = 0.0
    R = np.array([[0.1]]) * 10
    c0 = DiscreteQRCost(Q=np.zeros((model.state_size,model.state_size)), R=R)
    c0 = c0 + c0.goal_seeking_cost(Q=Q, x_target=model.x_upright)
    c0 = c0 + c0.goal_seeking_terminal_cost(xN_target=model.x_upright) * 1000
    return c0 * 2


class GymSinCosPendulumModel(DiscretizedModel): #!s=da #!s=da #!s=lec #!s=lec
    state_labels =  ['$\sin(\\theta)$', '$\cos(\\theta)$', '$\\dot{\\theta}$'] # Check if this escape character works.
    action_labels = ['Torque $u$']

    def __init__(self, dt=0.02, cost=None, transform_actions=True, **kwargs): #!s=da #!s=lec
        model = ContiniousPendulumModel(**kwargs) #!s=lec
        self.max_torque = model.max_torque
        self.transform_actions = transform_actions  #!s=da
        super().__init__(model=model, dt=dt, cost=cost) #!s=lec #!s=lec  #!s=da  #!s=da
        self.x_upright = np.asarray(self.continious_states2discrete_states(model.x_upright))
        self.l = model.l # Pendulum length
        if cost is None:  #!s=da
            cost = _pendulum_cost(self)
        self.cost = cost  #!s=da

    #!s=da
    def sym_discrete_xu2continious_xu(self, x, u):
        sin_theta, cos_theta, theta_dot = x[0], x[1], x[2]
        torque = sym.tanh(u[0]) * self.max_torque if self.transform_actions else u[0]
        theta = sym.atan2(sin_theta, cos_theta)  # Obtain angle theta from sin(theta),cos(theta)
        return [theta, theta_dot], [torque]

    def sym_continious_xu2discrete_xu(self, x, u): #!s=vartransform
        theta, theta_dot = x[0], x[1]
        torque = sym.atanh(u[0]/self.max_torque) if self.transform_actions else u[0]
        return [sym.sin(theta), sym.cos(theta), theta_dot], [torque] #!s=da  #!s=vartransform


class GymSinCosPendulumEnvironment(ContiniousTimeEnvironment): #!s=eb
    def __init__(self, *args, Tmax=5, supersample_trajectory=False, transform_actions=True, render_mode=None, **kwargs): #!s=eb
        discrete_model = GymSinCosPendulumModel(*args, transform_actions=transform_actions, **kwargs) #!s=eb
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(discrete_model.action_size,), dtype=float)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(discrete_model.state_size,), dtype=float)
        super().__init__(discrete_model, Tmax=Tmax, supersample_trajectory=supersample_trajectory, render_mode=render_mode) #!s=eb

    def step(self, u):
        self.discrete_model.continuous_model.u_prev = u
        return super().step(u)

if __name__ == "__main__":
    model = ContiniousPendulumModel(l=1, m=1)
    print(str(model))
    print(f"Pendulum with g={model.g}, l={model.l}, m={model.m}") #!o
    x = [1,2]
    u = [0] # Input state/action.
    x_dot = model.f([1, 2], [0], t=0) #!b # x_dot = ... #!b Compute dx/dt = f(x, u, t=0) here using the model-class defined above
    x_dot_numpy = model.f([1, 2], [0], t=0)  #!b # x_dot_numpy = ... #!b Compute dx/dt = f(x, u, t=0) here using numpy-expressions you write manually.

    print(f"Using model-class: dx/dt = f(x, u, t) = {x_dot}")
    print(f"Using numpy: dx/dt = f(x, u, t) = {x_dot_numpy}") #!o

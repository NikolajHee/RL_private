# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
import sympy as sym
from scipy.optimize import Bounds
from gym.spaces import Box
import matplotlib.pyplot as plt
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment
from irlc.ex04.continuous_time_model import ContiniousTimeSymbolicModel
from irlc.ex04.cost_continuous import SymbolicQRCost
from irlc.ex05.direct_agent import DirectAgent
from irlc.ex05.direct import get_opts
from irlc.ex06.linearization_agent import LinearizationAgent
from irlc.project2.utils import render_car
from irlc import Agent, train, plot_trajectory, savepdf#, VideoMonitor
#from irlc import VideoMonitor

dt = 0.05 # Time discretization Delta
Tmax = 5  # Total simulation time (in all instances). This means that N = Tmax/dt = 100.
x22 = (2, 2, np.pi / 2)  # Where we want to drive to: x_target

class R2D2Model(ContiniousTimeSymbolicModel): # This may help you get started.
    # TODO: 4 lines missing.
    state_size = 3
    action_size = 2
    state_labels = ["$x$", "$y$", r"$\gamma$"]
    action_labels = ["$v$", r"$\omega$"]
    #raise NotImplementedError("Define constants as needed here (look at other environments); Note there is an easy way to add labels!")

    def __init__(self, x_target=x22, Tmax=5., Q0=1., dt=0.05): # This constructor is one possible choice.
        bounds = {}  # Set this variable to correspond to the simple (linear) bounds the system is subject to. See exercises for examples.
        self.x_target = np.asarray(x_target)
        """ Since we have not discussed the cost-function a lot during the exercises it feels unreasonable to have you
        poke through the API. The following should do the trick: """
        cost = SymbolicQRCost(Q=np.zeros(self.state_size), R=np.eye(self.action_size))
        cost += cost.goal_seeking_cost(x_target=self.x_target)*Q0

        # TODO: 6 lines missing.
        self.Delta = dt
        #raise NotImplementedError("Complete model body.")

        bounds = dict(tF_low=Tmax, tF_high=Tmax,
                      x0_low=[0]*3,x0_high=[0]*3,
                      x_low=[-np.inf]*3, x_high=[np.inf]*3,
                      u_low=[-np.inf]*2, u_high=[np.inf]*2,
                      xF_low=x_target, xF_high=x_target
                      )
        # Set up a variable for rendering (optional) and call superclass.
        self.viewer = None
        super().__init__(cost=cost, bounds=bounds)

    # TODO: 6 lines missing.
    def sym_f(self, x, u, t=None): 
        #return [x[0] + self.Delta * u[0] * sym.cos(x[2]), x[1] + self.Delta * u[0] * sym.sin(x[2]), x[2] + self.Delta*u[1]]
        return [u[0] * sym.cos(x[2]), u[0] * sym.sin(x[2]), u[1]]
        
         # see model_cartpole

    #raise NotImplementedError("Complete model here.")

    """ These are two helper functions. They add rendering functionality so you can use the environment as
    > env = VideoMonitor(env) 
    and see a small animation. 
    """
    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def render(self, x, mode="human"): 
        return render_car(self, x, x_target=self.x_target, mode=mode) 

class R2D2DiscreteModel(DiscretizedModel):
    def __init__(self, dt=0.05, Q0=0., x_target=x22, Tmax=5.):
        model = R2D2Model(x_target=x_target, Q0=Q0, Tmax=Tmax, dt=dt)
        super().__init__(model=model, dt=dt)

class R2D2Environment(ContiniousTimeEnvironment):
    def __init__(self, Tmax=Tmax, Q0=0., x_target=x22, dt=0.5, render_mode=None):
        Discrete_model = R2D2DiscreteModel(Q0=Q0, x_target=x_target, Tmax=Tmax, dt=dt)
        super().__init__(discrete_model=Discrete_model, Tmax=Tmax, render_mode=render_mode)

# TODO: 9 lines missing.
#raise NotImplementedError("Your code here.")
from irlc.ex06.dlqr import LQR
class ILinearizationAgent(Agent):
    """ Implement the simple linearization procedure described in (Her23, Algorithm 23) which expands around a single fixed point. """
    def __init__(self, env, model, xbar_init=None, ubar_init=None):
        self.model = model
        self.N = 50 
        self.us = [ubar_init]
        # xp, A, B = model.f(xbar_init, ubar_init, k=0, compute_jacobian=True) 
        # d = xp - A @ xbar_init - B @ ubar_init 
        self.Q, self.q, self.R = self.model.cost.Q, self.model.cost.q, self.model.cost.R
        # (self.L, self.l), (V, v, vc) = LQR(A=[A]*N, B=[B]*N, d=[d]*N, Q=[Q]*N, q=[q]*N, R=[self.model.cost.R]*N) 
        super().__init__(env)

    def pi(self, x, k, info=None):
        xp, A, B = self.model.f(x, self.us[k], k=0, compute_jacobian=True) 
        d = xp - A @ x - B @ self.us[k]
        (L, l), (V, v, vc) = LQR(A=[A]*self.N, B=[B]*self.N, d=[d]*self.N, Q=[self.Q]*self.N, q=[self.q]*self.N, R=[self.R]*self.N) 
        u = L[0] @ x + l[0] 
        self.us.append(u)
        return u


def f_euler(x, u, Delta=0.1): 
    """ Solve Problem 13. The function should compute
    > x_next = f_k(x, u)
    """
    # TODO: 1 lines missing.
    #raise NotImplementedError("return next state")
    #x_next = np.array([x[0], x[1], x[2]]) + Delta * np.array([ u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])
    x_next = R2D2DiscreteModel(dt = Delta, Q0=1).f(x,u)
    return x_next

def linearize(x_bar, u_bar, Delta=0.05): 
    """ Linearize R2D2's dynamics around the two vectors x_bar, u_bar
    and return A, B, d so that

    x_{k+1} = A x_k + B u_k + d (approximately)

    assuming that x_k and u_k are close to x_bar, u_bar. The function should return A, B and d.
    """
    # TODO: 2 lines missing.
    model = R2D2DiscreteModel(dt = Delta)
    xp, A, B = model.f(x_bar, u_bar, k=0, compute_jacobian=True) 
    d = xp - A @ x_bar - B @ u_bar 
    #raise NotImplementedError("return A, B, d as numpy ndarrays.")
    return A, B, d

def drive_to_linearization(x_target, plot=True): 
    """
    Plan in a R2D2 model with specific value of x_target (in the cost function). We use Q0=1.0.

    this function will linearize the dynamics around xbar=0, ubar=0 to get a linear approximation of the model,
    and then use that to plan on a horizon of N=50 steps to get a control law (L_0, l_0). This is then applied
    to generate actions.

    Plot is an optional parameter to control plotting. the plot_trajectory(trajectory, env) method may be useful.

    The function should return the states visited as a (samples x state-dimensions) matrix, i.e. same format
    as the default output of trajectories when you use train(...).

    Hints:
        * The control method is identical to one we have seen in the exercises/notes. You can re-purpose the code from that week.
        * Remember to set Q0=1
    """
    # TODO: 8 lines missing.
    xbar = np.array([0,0,0])
    ubar = np.array([0,0])
    env = R2D2Environment(Q0 = 1.0, x_target=x_target, dt=dt)
    #model = R2D2DiscreteModel(Q0=1, x_target=x_target)
    agent = LinearizationAgent(env, env.discrete_model, xbar, ubar)
    _, traj = train(env, agent, num_episodes=1, return_trajectory=True)#, reset=False)
    if plot:
        plot_trajectory(traj[0], env)#, xkeys=[0,2, 3], ukeys=[0])
    #raise NotImplementedError("Implement function body")
    #env = VideoMonitor(env) 
    return traj[0].state

def drive_to_direct(x_target, plot=False): 
    """
    Optimal planning in the R2D2 model with specific value of x_target using the direct method.
    Remember that for this problem we set Q0=0, and implement x_target as an end-point constraint (see examples from exercises).

    Plot is an optional parameter to control plotting, and to (optionally) visualize the environment using code such as

    if plot:
        env = VideoMonitor(env)

    For making the actual plot, the plot_trajectory(trajectory, env) method may be useful (see examples from exercises to see how labels can be specified)

    The function should return the states visited as a (samples x state-dimensions) matrix, i.e. same format
    as the default output of trajectories when you use train(...).

    Hints:
        * The control method (Direct method) is identical to what we did in the exercises, but you have to specify the options
        to implement the correct grid-refinement of N=10, N=20 and N=40.
        * The guess()-function will be automatically specified correctly assuming you implement the correct bounds. Use that function in the options.
    """
    # TODO: 10 lines missing.
    #guess = {'t0': 0,
    #          'tF': 4,
    #          'x': [np.asarray([0, 0]), np.asarray([np.pi, 0])],
    #          'u': [np.asarray([0]), np.asarray([0])]}
    dmodel = R2D2DiscreteModel(dt = 0.05, Q0 = 0, x_target = x_target)
    #dmodel.continuous_model.bounds = {'tF_low': 0.5,                 'tF_high': 5,  
    #                 't0_low': 0,                   't0_high': 0,
    #                 'x_low': [-2 * np.pi, -np.inf],'x_high': [2 * np.pi, np.inf],
    #                 'u_low': [-max_torque],        'u_high': [max_torque],
    #                 'x0_low': [np.pi, 0],          'x0_high': [np.pi, 0],
    #                 'xF_low': [0, 0],              'xF_high': [0, 0] 
    #                 }
    env = ContiniousTimeEnvironment(discrete_model=dmodel, Tmax = Tmax)
    guess = env.discrete_model.continuous_model.guess()

    options = [get_opts(N=10, ftol=1e-3, guess=guess),
               get_opts(N=20, ftol=1e-3),#, guess=guess),
               get_opts(N=40, ftol=1e-3),#, guess=guess)
               ]

    agent = DirectAgent(env)#, options=options,guess=guess)
    env.Tmax = agent.solutions[-1]['fun']['tF'] # Specify max runtime of the environment. Must be based on the Agent's solution.
    stats, traj = train(env, agent=agent, num_episodes=1, return_trajectory=True)

    #raise NotImplementedError("Implement function body")
    return traj[0].state


def drive_to_mpc(x_target, plot=True): 
    """
    Plan in a R2D2 model with specific value of x_target (in the cost function) using iterative MPC (see text).
    In this problem, we set Q0=1.

    Plot is an optional parameter to control plotting. the plot_trajectory(trajectory, env) method may be useful.

    The function should return the states visited as a (samples x state-dimensions) matrix, i.e. same format
    as the default output of trajectories when you use train(...).

    Hints:
    * The control method is nearly identical to the linearization control method. Think about the differences,
    and how a solution to one can be used in another.
    """
    # TODO: 5 lines missing.
    #raise NotImplementedError("Implement function body")
    xbar = np.array([0,0,0])
    ubar = np.array([0,0])
    env = R2D2Environment(Q0 = 1.0, x_target=x_target, dt=dt)
    #model = R2D2DiscreteModel(Q0=1, x_target=x_target)
    agent = ILinearizationAgent(env, env.discrete_model, xbar, ubar)
    _, traj = train(env, agent, num_episodes=1, return_trajectory=True)#, reset=False)
    if plot:
        plot_trajectory(traj[0], env)#, xkeys=[0,2, 3], ukeys=[0])
    
    return traj[0].state

if __name__ == "__main__":
    # Check Problem 14
    x = np.asarray( [0, 0, 0] )
    u = np.asarray( [1,0])
    print("x_k =", x, "u_k =", u, "x_{k+1} =", f_euler(x, u, dt))

    A,B,d = linearize(x_bar=x, u_bar=u, Delta=dt)
    print("x_{k+1} ~ A x_k + B u_k + d")
    print("A:", A)
    print("B:", B)
    print("d:", d)

    # Test the simple linearization method (Problem 16)
    states = drive_to_direct(x22, plot=True)
    savepdf('r2d2_direct')
    plt.show()
    # Build plot assuming that states is in the format (samples x coordinates-of-state).
    plt.plot(states[:,0], states[:,1], 'k-', label="R2D2's (x, y) trajectory")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    savepdf('r2d2_direct_B')
    plt.show()

    # Test the simple linearization method (Problem 17)
    drive_to_linearization((2,0,0), plot=True)
    savepdf('r2d2_linearization_1')
    plt.show()

    drive_to_linearization(x22, plot=True)
    savepdf('r2d2_linearization_2')
    plt.show()

    # Test iterative LQR (Problem 18)
    state = drive_to_mpc(x22, plot=True)
    print(state[-1])
    savepdf('r2d2_iterative_1')
    plt.show()
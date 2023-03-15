# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import time
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg') # Matplotlib is having a bad year, and this will stop it from randomly crashing on some platforms (linux), but may cause problems on other (mac).
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment
from irlc.ex04.continuous_time_model import ContiniousTimeSymbolicModel
from irlc.ex04.cost_continuous import SymbolicQRCost
from irlc.ex05.direct_agent import DirectAgent
from irlc.ex05.direct import get_opts
from irlc.ex06.linearization_agent import LinearizationAgent
from irlc.project2.utils import R2D2Viewer
from irlc import Agent, train, plot_trajectory, savepdf

dt = 0.05 # Time discretization Delta
Tmax = 5  # Total simulation time (in all instances). This means that N = Tmax/dt = 100.
x22 = (2, 2, np.pi / 2)  # Where we want to drive to: x_target

class R2D2Model(ContiniousTimeSymbolicModel): # This may help you get started.
    # TODO: 4 lines missing.
    raise NotImplementedError("Define constants as needed here (look at other environments); Note there is an easy way to add labels!")

    def __init__(self, x_target=(2,2,np.pi/2), Tmax=5., Q0=1.): # This constructor is one possible choice.
        bounds = {}  # Set this variable to correspond to the simple (linear) bounds the system is subject to. See exercises for examples.
        self.x_target = np.asarray(x_target)
        """ Since we have not discussed the cost-function a lot during the exercises it feels unreasonable to have you
        poke through the API. The following should do the trick: """
        cost = SymbolicQRCost(Q=np.zeros(self.state_size), R=np.eye(self.action_size))
        cost += cost.goal_seeking_cost(x_target=self.x_target)*Q0
        # Specify bounds
        # TODO: 6 lines missing.
        raise NotImplementedError("Insert your solution and remove this error.")
        # Set up a variable for rendering (optional) and call superclass.
        self.viewer = None
        super().__init__(cost=cost, bounds=bounds)

    # TODO: 3 lines missing.
    raise NotImplementedError("Complete model here.")

    """ These are two helper functions. They add rendering functionality so you can eventually use the environment as
    
    > env = R2D2Environment(render_mode='human') 
    
    and see a small animation. 
    """
    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def render(self, x, render_mode="human"): 
        if self.viewer is None:
            self.viewer = R2D2Viewer(x_target=self.x_target) # Target is the red cross.
        self.viewer.update(x)
        time.sleep(0.05)
        return self.viewer.blit(render_mode=render_mode) 

class R2D2DiscreteModel(DiscretizedModel):
    def __init__(self, dt=0.05, Q0=0., x_target=x22, Tmax=5.):
        super().__init__(model=R2D2Model(x_target=x_target, Q0=Q0, Tmax=Tmax), dt=dt)

class R2D2Environment(ContiniousTimeEnvironment):
    def __init__(self, Tmax=Tmax, Q0=0., x_target=x22, dt=0.5, render_mode=None):
        super().__init__(R2D2DiscreteModel(Q0=Q0, x_target=x_target, Tmax=Tmax, dt=dt), Tmax=Tmax, render_mode=render_mode)

# TODO: 9 lines missing.
raise NotImplementedError("Your code here.")

def f_euler(x : np.ndarray, u : np.ndarray, Delta=0.05) -> np.ndarray: 
    """ Solve Problem 13. The function should compute
    > x_next = f_k(x, u)
    """
    # TODO: 1 lines missing.
    raise NotImplementedError("return next state")
    return x_next

def linearize(x_bar, u_bar, Delta=0.05):
    """ Linearize R2D2's dynamics around the two vectors x_bar, u_bar
    and return A, B, d so that

    x_{k+1} = A x_k + B u_k + d (approximately)

    assuming that x_k and u_k are close to x_bar, u_bar. The function should return linearization matrices A, B and d.
    """
    # return A, B, d as numpy ndarrays.
    # TODO: 2 lines missing.
    raise NotImplementedError("Insert your solution and remove this error.")
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
    # TODO: 9 lines missing.
    raise NotImplementedError("Implement function body")
    return traj[0].state

def drive_to_direct(x_target, plot=False): 
    """
    Optimal planning in the R2D2 model with specific value of x_target using the direct method.
    Remember that for this problem we set Q0=0, and implement x_target as an end-point constraint (see examples from exercises).

    Plot is an optional parameter to control plotting, and to (optionally) visualize the environment using code such as

    if plot:
        env = R2D2Environment(render_mode='human')

    For making the actual plot, the plot_trajectory(trajectory, env) method may be useful (see examples from exercises to see how labels can be specified)

    The function should return the states visited as a (samples x state-dimensions) matrix, i.e. same format
    as the default output of trajectories when you use train(...).

    Hints:
        * The control method (Direct method) is identical to what we did in the exercises, but you have to specify the options
        to implement the correct grid-refinement of N=10, N=20 and N=40.
        * In the first iteration, you need a guess. The guess()-function in the ContiniousTimeSymbolicModel is automatically specified correctly assuming
          you implement the correct bounds. Use that function in the options (see exercises).
    """
    # TODO: 11 lines missing.
    raise NotImplementedError("Implement function body")
    return traj[0].state


def drive_to_mpc(x_target, plot=True) -> np.ndarray: 
    """
    Plan in a R2D2 model with specific value of x_target (in the cost function) using iterative MPC (see problem text).
    Use Q0 = 1. in the cost function (see the R2D2 model class)

    Plot is an optional parameter to control plotting. the plot_trajectory(trajectory, env) method may be useful.

    The function should return the states visited as a (samples x state-dimensions) matrix, i.e. same format
    as the default output of trajectories when you use train(...).

    Hints:
     * The control method is *nearly* identical to the linearization control method. Think about the differences,
       and how a solution to one can be used in another.
     * A bit more specific: Linearization is handled similarly to the LinearizationAgent, however, we need to update
       (in each step) the xbar/ubar states/actions we are linearizing about, and then just use the immediate action computed
       by the linearization agent.
     * My approach was to implement a variant of the LinearizationAgent.
    """
    # TODO: 6 lines missing.
    raise NotImplementedError("Implement function body")
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

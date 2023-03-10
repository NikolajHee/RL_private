# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import matplotlib.pyplot as plt
import numpy as np
from irlc import savepdf, train, plot_trajectory
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment
from irlc.ex04.model_linear_quadratic import LinearQuadraticModel
from irlc.ex04.pid_locomotive_agent import PIDLocomotiveAgent
from irlc.ex06.lqr_agent import DiscreteLQRAgent

g = 9.82      # Gravitational force in m/s^2 
m = 0.1       # Mass of pendulum in kg
Tmax = 8      # Planning horizon in seconds
Delta = 0.04  # Time discretization constant 

# TODO: 21 lines missing.
#raise NotImplementedError("Your solution here")
class pid_pend:
    def __init__(self, ):
        pass
    def _get_initial_state(self):
        return np.asarray([1, 0])


def get_A_B(g, L, m=0.1): 
    """ Compute the two matrices A, B (see Problem 1) here and return them.
    The matrices should be numpy ndarrays. """
    # TODO: 2 lines missing.
    A = np.array([[0, 1], [-g/L,0]])
    B = np.array([0,1/(m*L**2)]).T
    #raise NotImplementedError("Compute A and B here")
    return A, B

def cost_discrete(x_k, u_k): 
    """ Compute the (dicretized) cost at time k given x_k, u_k. in the Yoda-problem.
    I.e. the total cost is

    > Cost = sum_{k=0}^{N-1} c_k(x_k, u_k)

    and this function should return c_k(x_k, u_k).

    The idea behind asking you to implement this function is to test you have the right cost-function, as
    otherwise the code can be fairly hard to debug. If you are following the framework, you can implement the function
    using commands such as:

    dmodel = (create a discrete model instance here)
    return dmodel.cost.c(x_k, u_k)

    If this worked, you will know you implemented the R, Q matrices correctly.
    """
    # TODO: 2 lines missing.
    raise NotImplementedError("Implement function body")
    return c_k

def problem1(L): 
    """ This function solve Problem 2 by defining a PID controller and making the plot. The recommended way to do this
    is by implementing a very simple environment corresponding to Yodas pendulum (note we have several examples of linear models, including
    the Harmonic Osscilator, see model_harmonic.py), and then use an appropriate agent on it to simulate the PID controller.

    Hints:
        * You can perhaps be inspired by the locomotive-problem
        * The plot_trajectory(trajectory, environment) function is plenty fine for making this kind of plot. As an example:

    plot_trajectory(traj[0], env)
    plt.title("PID agent heuristic")
    savepdf("yoda1")
    plt.show()
    """
    # TODO: 7 lines missing.
    raise NotImplementedError("Implement function body")

def part1(L): 
    """ This function solve Problem 3.
    It should solve the Pendulum problem using an optimal LQ control law and return L_0, l_0 as well as the action-sequence
    obtained when controlling the system using this exact control law at all time instances k.

    Hints:
        * Although we don't have an agent that does *exact* what we want in the problem, we have one that comes *really* close.
    """
    # TODO: 3 lines missing.
    raise NotImplementedError("Return L0, l0, and the action sequence from the LQR controller")
    return agent.L[0], agent.l[0], traj[0].action

def part2(L): 
    """ This function should solve Problem 4. The function should return the action sequence
    obtained by treating the LQR control law as the parameters for a PID controller, and then simulating the system
    using that PID controller.

    The function should return

    > K_P, K_I, K_D, x_star, action_sequence

    in that order, where the first 4 terms specify a PID controller and the last is the corresponding action-sequence
    obtained by simulating the controller. The simulation can be done using the same method you used to
    simulate your pid controller.

    Hints:
        * Once you have specified the PID controller parameters, you can probably re-use what you did for Problem 1.

    """
    # TODO: 6 lines missing.
    raise NotImplementedError("return x*, Kp, Kd, Ki, and the action sequence from the PID controller")
    return Kp, Ki, Kd, x_star, traj[0].action

if __name__ == "__main__":
    L = 0.4  # Length of pendulum string; sorry that this clashes with (L_k, l_k)

    # Solve Problem 2
    problem1(L)

    # Optimal LQR to stop the pendulum Problem 3. Print control law and action sequence.
    L0, l0, u1 = part1(L)
    print(L0, l0)
    # Print the cost_term at time k, c_k(x_k, u_k), for a state/action:
    print("Cost at time k is c_k(x_k, u_k) =", cost_discrete(np.asarray([-.1, 1]), np.asarray([-2])))

    # Solve Problem 4
    Kp, Ki, Kd, x_star, u2 = part2(L)
    print(x_star, Kp, Kd, Ki)

    # Plot the two action sequences (Problem 5)
    plt.plot(np.linspace(0, Tmax - Delta, len(u2)), u2, label="PID action sequence")
    plt.plot(np.linspace(0, Tmax - Delta, len(u1)), u1, label="LQR Optimal action sequence")
    plt.legend()
    savepdf("yoda1_actions")
    plt.show()

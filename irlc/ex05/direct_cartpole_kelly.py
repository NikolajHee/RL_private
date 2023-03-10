# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Kel17] Matthew Kelly. An introduction to trajectory optimization: how to do your own direct collocation. SIAM Review, 59(4):849–904, 2017. (See kelly2017.pdf).
"""
from irlc.ex04.model_cartpole import ContiniousCartpole
from irlc.ex04.cost_continuous import SymbolicQRCost
from irlc.ex05.direct import direct_solver, get_opts
import numpy as np

def make_cartpole_kelly17():
    """
    Creates Cartpole problem. Details about the cost function can be found in (Kel17, Section 6)
    and details about the physical parameters can be found in (Kel17, Appendix E, table 3).
    """
    # this will generate a different carpole environment with an emphasis on applying little force u.
    duration = 2.0
    cost = None # Define the cost as a SymbolicQRCost(Q=..., R=...)
    # TODO: 2 lines missing.
    raise NotImplementedError("Insert your solution and remove this error.")
    # Initialize the cost-function above. You should do so by using a call of the form:
    # cost = SymbolicQRCost(Q=..., R=...) # Take values from Kelly
    # The values of Q, R can be taken from the paper.

    # _, sbounds, _, bounds = kelly_swingup(maxForce=20, dist=1.0) # get a basic version of the bounds (then update them below).
    maxForce = 20
    dist = 1.0
    bounds = dict(x_low=[-2 * dist, -np.inf, -2 * np.pi, -np.inf], x_high=[2 * dist, np.inf, 2 * np.pi, np.inf],
                  u_low=[-maxForce], u_high=[maxForce],
                  x0_low=[0, 0, np.pi, 0], x0_high=[0, 0, np.pi, 0],
                  xF_low=[dist, 0, 0, 0], xF_high=[dist, 0, 0, 0])

    # TODO: 2 lines missing.
    raise NotImplementedError("Update the bounds so the problem will take exactly tF=2 seconds.")

    # Instantiate the environment as a ContiniousCartpole model. The call should be of the form:
    # model = ContiniousCartpole(...)
    # Make sure you supply all relevant physical constants (maxForce, mp, mc, g and l) as well as the cost and bounds. Check the
    # ContiniousCartpole class definition for details.
    # TODO: 1 lines missing.
    raise NotImplementedError("Insert your solution and remove this error.")
    guess = model.guess()
    guess['tF'] = duration # Our guess should match the constraints.
    return model, guess

def compute_solutions():
    model, guess = make_cartpole_kelly17()
    print("cartpole mp", model.mp)
    options = [get_opts(N=10, ftol=1e-3, guess=guess),
               get_opts(N=40, ftol=1e-6)]
    solutions = direct_solver(model, options)
    return model, solutions

def direct_cartpole():
    model, solutions = compute_solutions()
    from irlc.ex05.direct_plot import plot_solutions
    print("Did we succeed?", solutions[-1]['solver']['success'])
    plot_solutions(model, solutions, animate=True, pdf="direct_cartpole_force")
    model.close()

if __name__ == "__main__":
    direct_cartpole()

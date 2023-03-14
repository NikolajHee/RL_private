# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex04.model_cartpole import ContiniousCartpole
from irlc.ex05.direct import direct_solver, get_opts
from irlc.ex05.direct_plot import plot_solutions
from irlc.ex04.cost_continuous import SymbolicQRCost
import numpy as np

def compute_solutions():
    """
    See: https://github.com/MatthewPeterKelly/OptimTraj/blob/master/demo/cartPole/MAIN_minTime.m
    """
    # Define the bounds here.

    maxForce = 50   # -50 <= u <= 50
    dist = 1        # Distance to traverse (i.e., x_1(tF) = dist)
    bounds = dict(tF_low=0.01, tF_high=np.inf,
                  u_low=[-maxForce], u_high=[maxForce],
                  )
    # Add missing bounds below. Add x_low/high, x0_low/high, xF_low/high.
    # TODO: 5 lines missing.
    raise NotImplementedError("Define bounds here corresponding to the minimum-time swingup problem.")

    cost = SymbolicQRCost(R=np.eye(1)*0, Q=np.eye(4)*0, qc=1)  # just minimum time
    model = ContiniousCartpole(maxForce=50, mp=0.5, mc=2.0, g=9.81, l=0.5, cost=cost, bounds=bounds)
    options = [get_opts(N=8, ftol=1e-3, guess=model.guess()),
               get_opts(N=16, ftol=1e-6),                # This is a hard problem and we need gradual grid-refinement.
               get_opts(N=32, ftol=1e-6),
               get_opts(N=70, ftol=1e-6)
               ]
    solutions = direct_solver(model, options)
    return model, solutions

if __name__ == "__main__":
    model, solutions = compute_solutions()
    x_sim, u_sim, t_sim = plot_solutions(model, solutions[:], animate=True, pdf="direct_cartpole_mintime")
    model.close()
    print("Did we succeed?", solutions[-1]['solver']['success'])

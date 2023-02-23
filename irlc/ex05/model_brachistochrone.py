# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
The Brachistochrone problem. See
https://apmonitor.com/wiki/index.php/Apps/BrachistochroneProblem
and (Bet10)
References:
  [Bet10] John T Betts. Practical methods for optimal control and estimation using nonlinear programming. Volume 19. Siam, 2010.
"""
from gym.spaces import Box
from scipy.optimize import Bounds
import sympy as sym
import numpy as np
from irlc.ex04.continuous_time_model import ContiniousTimeSymbolicModel
from irlc.ex04.cost_continuous import SymbolicQRCost

class ContiniouBrachistochrone(ContiniousTimeSymbolicModel): 
    state_labels= ["$x$", "$y$", "bead speed"]
    action_labels = ['Tangent angle']

    def __init__(self, g=9.82, h=None, x_dist=1, simple_bounds=None, cost=None): 
        self.g = g
        self.h = h
        self.x_dist = x_dist # or x_B

        guess0 = {'t0': 0,
                 'tF': 2,
                 'x': [np.asarray([0, 0, 0]), np.asarray([x_dist, x_dist, 2])],
                 'u': [np.asarray([0]), np.asarray([1])]}

        bounds = dict(x_low=[-np.inf, -np.inf, -np.inf], x_high=[np.inf, np.inf, np.inf],
                      u_low=[-np.inf], u_high=[np.inf],
                      x0_low=[0, 0, 0], x0_high=[0, 0, 0],
                      xF_low=[x_dist, -np.inf, -np.inf], xF_high=[x_dist, np.inf, np.inf],
                      )

        if cost is None:
            # TODO: 1 lines missing.
            raise NotImplementedError("Instantiate cost=SymbolicQRCost(...) here corresponding to minimum time.")

        super().__init__(cost=cost, bounds=bounds)


    def sym_f(self, x, u, t=None): 
        # TODO: 3 lines missing.
        raise NotImplementedError("Implement function body")
        return xp

    def sym_h(self, x, u, t):
        '''
        Add a dynamical constraint of the form

        h(x, u, t) <= 0
        '''
        if self.h is None:
            return []
        else:
            # compute a single dynamical constraint as in (Bet10, Example (4.10)) (Note y-axis is reversed in the example)
            # TODO: 1 lines missing.
            raise NotImplementedError("Insert your solution and remove this error.")

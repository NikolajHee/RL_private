# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import sympy as sym
from irlc.ex04.continuous_time_model import ContiniousTimeSymbolicModel
from irlc.ex04.cost_continuous import SymbolicQRCost
import numpy as np

class LinearQuadraticModel(ContiniousTimeSymbolicModel):
    """
    Implements a model with update equations

    dx/dt = Ax + Bx + d
    Cost = integral_0^{t_F} (1/2 x^T Q x + 1/2 u^T R u + q' x + qc) dt
    """
    def __init__(self, A, B, Q, R, q=None, qc=None, d=None):  
        cost = SymbolicQRCost(R=R, Q=Q, q=q, qc=qc)
        self.A, self.B, self.d = A, B, d
        n, d = A.shape[0], B.shape[1]
        bounds = dict(x0_low=[0]*n, x0_high=[0]*n,
                      x_low=[-np.inf]*A.shape[0], x_high=[np.inf]*A.shape[0],
                      u_low=[-np.inf]*B.shape[1], u_high=[np.inf]*B.shape[1])
        super().__init__(cost=cost, bounds=bounds)

    def sym_f(self, x, u, t=None):  
        xp = sym.Matrix(self.A) * sym.Matrix(x) + sym.Matrix(self.B) * sym.Matrix(u)
        if self.d is not None:
            xp += sym.Matrix(self.d)
        return [x for xr in xp.tolist() for x in xr]  

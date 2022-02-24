# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.
"""
from irlc.ex04.continuous_time_model import symv
import sympy as sym
import numpy as np
from irlc.ex04.continuous_time_model import ensure_policy
# Patch sympy with mapping to numpy functions.
sympy_modules_ = ['numpy', {'atan': np.arctan, 'atan2': np.arctan2, 'atanh': np.arctanh}, 'sympy']
import sys

class DiscretizedModel: 
    """
    A discretized model. To create a model of this type, first specify a symbolic model, then pass it along to the constructor.
    Since the symbolic model will specify the dynamics as a symbolic function, the discretized model can automatically discretize it
    and create functions for computing derivatives.

    It will also discretize the cost. Not it is possible to specify coordinate transformations.

    """
    state_labels = None
    action_labels = None

    def __init__(self, model, dt, cost=None, discretization_method=None): 
        self.dt = dt
        self.continuous_model = model   
        if discretization_method is None:
            from irlc.ex04.model_linear_quadratic import LinearQuadraticModel
            if isinstance(model, LinearQuadraticModel):
                discretization_method = 'Ei'
            else:
                discretization_method = 'Euler'
        self.discretization_method = discretization_method.lower()


        if not hasattr(self, "action_space"):
            self.action_space = model.action_space
        if not hasattr(self, "observation_space"):
            self.observation_space = model.observation_space

        """ Initialize symbolic variables representing inputs and actions. """
        uc = symv("uc", model.action_size) 
        xc = symv("xc", model.state_size)
        xd, ud = self.sym_continious_xu2discrete_xu(xc, uc)

        u = symv("u", len(ud)) 
        x = symv('x', len(xd))
        """ y is a symbolic variable representing y = f(xs, us, dt) """
        x_next = self.f_discrete_sym(x, u, dt=dt) 
        """ compute the symbolic derivate of y wrt. z = (x,u): dy/dz """

        dy_dz = sym.Matrix([[sym.diff(f, zi) for zi in list(x) + list(u)] for f in x_next])
        """ Define (numpy) functions giving next state and the derivatives """
        self.f_z = sym.lambdify((tuple(x), tuple(u)), dy_dz, modules=sympy_modules_)
        # Create a numpy function corresponding to the discretized model x_{k+1} = f_discrete(x_k, u_k) 
        self.f_discrete = sym.lambdify((tuple(x), tuple(u)), x_next, modules=sympy_modules_)  
        # Make action/state transformation
        xc_, uc_ = self.sym_discrete_xu2continious_xu(x, u)
        self.discrete_states2continious_states = sym.lambdify( (x,), xc_, modules=sympy_modules_) # probably better to make these individual
        self.discrete_actions2continious_actions = sym.lambdify( (u,), uc_, modules=sympy_modules_)  # probably better to make these individual

        xd, ud = self.sym_continious_xu2discrete_xu(xc, uc)
        self.continious_states2discrete_states = sym.lambdify((xc,), xd, modules=sympy_modules_)
        self.continious_actions2discrete_actions = sym.lambdify((uc,), ud, modules=sympy_modules_)

        # set labels
        if self.state_labels is None:
            self.state_labels = self.continuous_model.state_labels

        if self.action_labels is None:
            self.action_labels = self.continuous_model.action_labels

        # Setup cost function
        if cost is None:
            self.cost = model.cost.discretize(env=self, dt=dt)
        else:
            self.cost = cost

    @property
    def state_size(self):
        return self.observation_space.low.size

    @property
    def action_size(self):
        return self.action_space.low.size

    def reset(self):
        x = self.continuous_model.reset()
        return np.asarray(self.continious_states2discrete_states(x))

    def f_discrete_sym(self, xs, us, dt):
        """ Discretize dx/dt = f(x,u,t) """
        xc, uc = self.sym_discrete_xu2continious_xu(xs, us)
        if self.discretization_method == 'euler':
            xdot = self.continuous_model.sym_f(x=xc, u=uc)
            xnext = [x_ + xdot_ * dt for x_, xdot_ in zip(xc, xdot)]
        elif self.discretization_method == 'ei':  # Assume the continuous model is linear; a bit hacky, but use exact Exponential integration in that case
            A = self.continuous_model.A
            B = self.continuous_model.B
            d = self.continuous_model.d
            """ These are the matrices of the continuous-time problem.
            > dx/dt = Ax + Bu + d
            and should be discretized using the exact integration technique (see (Her21, Subsection 11.2.3) and (Her21, Subsection 11.2.6)); 
            the precise formula you should implement is given in (Her21, eq. (11.45))
            
            Remember the output matrix should be symbolic (see Euler integration for examples) but you can assume there are no variable transformations for simplicity.            
            """
            from scipy.linalg import expm, inv  # These might be of use.
            """
            expm computes the matrix exponential: 
            > expm(A) = exp(A) 
            inv computes the inverse of a matrix inv(A) = A^{-1}.
            """
            Ad = expm(A * dt)
            n = Ad.shape[0]
            d =  d.reshape( (len(B),1) ) if d is not None else np.zeros( (n, 1) )
            Bud = B @ sym.Matrix(uc) + (sym.zeros(len(B),1) if d is None else d)
            x_next = sym.Matrix(Ad) @ sym.Matrix(xc) + dt * phi1(A * dt) @ Bud
            xnext = list(x_next)
        else:
            raise Exception("Unknown discreetization method", self.discretization_method)
        xnext, _ = self.sym_continious_xu2discrete_xu(xnext, uc)
        return xnext

    def simulate(self, x0, policy, t0, tF, N=1000):
        policy3 = lambda x, t: self.discrete_actions2continious_actions(ensure_policy(policy)(x, t))
        x, u, t = self.continuous_model.simulate(self.discrete_states2continious_states(x0), policy3, t0, tF, N_steps=N, method='rk4')
        # transform these
        xd = np.stack( [np.asarray(self.continious_states2discrete_states(x_)).reshape((-1,)) for x_ in x ] )
        ud = np.stack( [np.asarray(self.continious_actions2discrete_actions(u_)).reshape((-1,))  for u_ in u] )
        return xd, ud, t

    def f(self, x, u, k, compute_jacobian=False, compute_hessian=False): 
        """
        By defult this functions returns f_k(x,u).

        If compute_jacobian=True it will return the derivatives as well:
        > f_k(x,u), df_k(x,u)/dx, df_k(x,u)/du.
        Code currently contains a stub for computing Hessians, but this is not implemented at the moment.
        """
        fx = np.asarray( self.f_discrete(x, u) )
        if compute_jacobian:
            J = self.f_z(x, u)
            if compute_hessian:
                raise Exception("Not implemented")
                f_xx, f_ux, f_uu = None,None,None  # Not implemented.
                return fx, J[:, :self.state_size], J[:, self.state_size:], f_xx, f_ux, f_uu
            else:
                return fx, J[:, :self.state_size], J[:, self.state_size:]
        else:
            return fx 

    def c(self, x, u, i=None, compute_gradients=False): 
        v = self.cost.c(x, u, i)
        return v[0] if not compute_gradients else v 

    def cN(self, x, compute_gradients=False):  
        v = self.cost.cN(x)
        return v[0] if not compute_gradients else v  

    def render(self, x=None, mode="human"):
        return self.continuous_model.render(x=self.discrete_states2continious_states(x), mode=mode)

    def sym_continious_xu2discrete_xu(self, x, u):
        """
        Handle coordinate transformations in the environment.
        Given x and u as (symbolic) expressions, the function computes:

        > x_k = phi_x(x)
        > u_k = phi_u(u)

        and return u_k, x_k
        """
        return x, u

    def sym_discrete_xu2continious_xu(self, x, u):
        """ Computes the inverse of the above function. I.e. returns
        > phi_x^{-1}(x_k)
        > phi_u^{-1}(u_k)
        """
        return x, u

    def close(self):
        self.continuous_model.close()


def phi1(A):
    from scipy.linalg import expm
    from math import factorial
    if np.linalg.cond(A) < 1 / sys.float_info.epsilon:
        return np.linalg.solve(A, expm(A) - np.eye( len(A) ) )
    else:
        C = np.zeros_like(A)
        for k in range(1, 20):
            dC = np.linalg.matrix_power(A, k - 1) / factorial(k)
            C += dC
        assert sum( np.abs(dC.flat)) < 1e-10
        return C

# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her23] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2023.
"""
from irlc.ex04.continuous_time_model import symv, ContiniousTimeSymbolicModel
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

    The class will also discretize the cost. Note that it is possible to specify coordinate transformations.
    """
    state_labels = None
    action_labels = None

    "This field represents the :class:`~irlc.ex04.continuous_time_model.ContinuousSymbolicModel` the discrete model is derived from."
    continuous_model = None

    def __init__(self, model : ContiniousTimeSymbolicModel, dt: float, cost=None, discretization_method=None): 
        """
        Create the discretized model.

        :param model: The continuous-time model to discretize.
        :param dt: Discretization timestep :math:`\Delta`
        :param cost: If this parameter is not specified, the cost will be derived (discretized) automatically from ``model``
        :param discretization_method: Can be either ``'Euler'`` (default) or ``'ei'`` (exponential integration). The later will assume that the model is a linear.
        """
        self.dt = dt  
        self.continuous_model = model   
        if discretization_method is None:
            from irlc.ex04.model_linear_quadratic import LinearQuadraticModel
            if isinstance(model, LinearQuadraticModel):
                discretization_method = 'Ei'
            else:
                discretization_method = 'Euler'
        self.discretization_method = discretization_method.lower()

        """ Initialize symbolic variables representing inputs and actions. """
        uc = symv("uc", model.action_size) 
        xc = symv("xc", model.state_size)
        xd, ud = self.sym_continious_xu2discrete_xu(xc, uc)

        u = symv("u", len(ud)) 
        x = symv('x', len(xd))
        """ x_next is a symbolic variable representing x_{k+1} = f_k(x_k, u_k) """
        x_next = self.f_discrete_sym(x, u, dt=dt) 
        """ compute the symbolic derivate of x_next wrt. z = (x,u): d x_{k+1}/dz """
        dy_dz = sym.Matrix([[sym.diff(f, zi) for zi in list(x) + list(u)] for f in x_next])
        """ Define (numpy) functions giving next state and the derivatives """
        self.f_z = sym.lambdify((tuple(x), tuple(u)), dy_dz, modules=sympy_modules_) 
        # Create a numpy function corresponding to the discretized model x_{k+1} = f_discrete(x_k, u_k) 
        self.f_discrete = sym.lambdify((tuple(x), tuple(u)), x_next, modules=sympy_modules_)  
        self._n = len(x)
        self._d = len(u)

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
            self.cost = model.cost.discretize(dt=dt) 
        else:
            self.cost = cost

    @property
    def state_size(self):
        """
        The dimension of the state vector :math:`x`, i.e. :math:`n`
        :return: Dimension of the state vector :math:`n`
        """
        return self._n

    @property
    def action_size(self):
        """
        The dimension of the action vector :math:`u`, i.e. :math:`d`
        :return: Dimension of the action vector :math:`d`
        """
        return self._d

    def f_discrete_sym(self, xs, us, dt):
        """
        This is a helper function. It computes the discretized dynamics as a symbolic object:

        .. math::
            x_{k+1}  = f_k(x_k, u_k, t_k)

        The parameters corresponds to states and actions and are lists of the form ``[x0, x1, ..]`` and ``[u0, u1, ..]``
        where each element is a symbolic expression. The function returns a list of the form ``[f0, f1, ..]`` where
        each element is a symbolic expression corresponding to a coordinate of :math:`f_k`.

        :param xs: List of symbolic expressions corresponding to the coordinates of :math:`x_k`
        :param us: List of symbolic expressions corresponding to the coordinates of :math:`x_u`
        :param dt: A symbolic expressions corresponding to :math:`t_k`
        :return: A list of symbolic expressions corresponding to the coordinates of :math:`f_k`
        """
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
            and should be discretized using the exact integration technique (see (Her23, Subsection 11.2.3) and (Her23, Subsection 11.2.6)); 
            the precise formula you should implement is given in (Her23, eq. (11.45))
            
            Remember the output matrix should be symbolic (see Euler integration for examples) but you can assume there are no variable transformations for simplicity.            
            """
            from scipy.linalg import expm, inv
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
        # transform to discrete representations using phi.
        xd = np.stack( [np.asarray(self.continious_states2discrete_states(x_)).reshape((-1,)) for x_ in x ] )
        ud = np.stack( [np.asarray(self.continious_actions2discrete_actions(u_)).reshape((-1,))  for u_ in u] )
        return xd, ud, t

    def f(self, x, u, k=0, compute_jacobian=False): 
        """
        This function implements the dynamics :math:`f_k(x_k, u_k)` of the model. They can be evaluated as:

        .. runblock:: pycon

            >>> from irlc.ex04.model_pendulum import GymSinCosPendulumModel
            >>> model = GymSinCosPendulumModel()
            >>> x = [0, 1, 0.4]
            >>> u = [1]
            >>> print(model.f(x,u) )           # Computes x_{k+1} = f_k(x_k, u_k)

        The function can also compute erivatives of the dynamics by calling the function with
        ``compute_jacobian=True``. In this case the function will then return three arguments, corresponding to the
        usual dynamics and the Jacobian derives with respect to :math:`x` and :math:`u`:

        .. math::
            f_k(x,u), \quad J_x f_k(x,u), \quad J_u f_k(x, u)

        .. runblock:: pycon

            >>> from irlc.ex04.model_pendulum import GymSinCosPendulumModel
            >>> model = GymSinCosPendulumModel()
            >>> x = [0, 1, 0.4]
            >>> u = [0]
            >>> f, Jx, Ju = model.f(x,u, compute_jacobian=True)
            >>> print("Jacobian Jx is\\n", Jx)
            >>> print("Jacobian Ju is\\n", Ju)


        :param x: The state as a numpy array
        :param u: The action as a numpy array
        :param k: The time step as an integer (currently this has no effect)
        :param compute_jacobian: ``True`` if the function should return Jacobians
        :return:
            - ``f`` - The value of :math:`f_k(x,u)`
            - ``Jx`` - The Jacobian :math:`J_x f_k(x,u)` (as a matrix; only if ``compute_jacobian=True``)
            - ``Ju`` - The value of :math:`J_u f_k(x,u)` (as a matrix; only if ``compute_jacobian=True``)
        """
        fx = np.asarray( self.f_discrete(x, u) ) 
        if compute_jacobian:
            J = self.f_z(x, u)
            return fx, J[:, :self.state_size], J[:, self.state_size:]
        else:
            return fx  

    def render(self, x=None, render_mode="human"):
        return self.continuous_model.render(x=self.discrete_states2continious_states(x), render_mode=render_mode)

    def sym_continious_xu2discrete_xu(self, x, u):
        """
        This (optional) function handle coordinate transformations.
        ``x`` and ``u`` are lists of symbolic expressions (the state and action), and the function then computes and return
        the forward coordinate transformation (from continuous coordinates to discrete):

        .. math::
            x_k & = \phi_x(x) \\\\
            u_k & = \phi_u(u)

        :param x: Continuous state
        :param u: Continuous action
        :return:
            - ``x_k`` - Transformed (discrete) state
            - ``u_k`` - Transformed (discrete) action
        """
        return x, u

    def sym_discrete_xu2continious_xu(self, x_k, u_k):
        """
        This (optional) function handle coordinate transformations.
        ``x_k`` and ``u_k`` are lists of symbolic expressions (the state and action), and the function then computes and return
        the **backward** coordinate transformation (from discrete coordinates to continuous coordinates):

        .. math::
            x & = \phi^{-1}_x(x_k) \\\\
            u & = \phi^{-1}_u(u_k)

        :param x_k: discrete state
        :param u_k: discrete action
        :return:
            - ``x`` - Transformed (Continuous) state
            - ``u`` - Transformed (Continuous) action
        """
        return x_k, u_k

    def close(self):
        self.continuous_model.close()

    def __str__(self):
        """
        Return a string representation of the model. This is a potentially helpful way to summarize the content of the
        model. You can use it as:

        .. runblock:: pycon

            >>> from irlc.ex04.model_pendulum import GymSinCosPendulumModel
            >>> model = GymSinCosPendulumModel()
            >>> print(model)

        :return: A string containing the details of the model.
        """
        split = "-"*20
        s = [f"{self.__class__}"] + ['='*50]
        s += [f"Dynamics (after discretization with Delta = {self.dt}):", split]
        t = sym.symbols("t")
        x = symv("x", self.state_size)
        u = symv("u", self.action_size)
        s += [f"f_k({x}, {u}) = {str(self.f_discrete_sym(x, u, self.dt))}", '']
        s += ["Continuous-time dynamics:", split]
        xc = symv("x", self.continuous_model.state_size)
        uc = symv("u", self.continuous_model.action_size)
        s += [f"f_k({x}, {u}) = {str(self.continuous_model.sym_f(xc, uc))}", '']
        s += ["Variable transformations:", split]
        # self.continious_states2discrete_states(xc)
        xd, ud = self.sym_continious_xu2discrete_xu(xc, uc)
        s += [f' * phi_x( x(t) ) -> x_k = {xd}']
        s += [f' * phi_u( u(t) ) -> u_k = {ud}', '']
        s += ["Cost:", split, str(self.cost)]
        return "\n".join(s)


def phi1(A):
    """ This is a helper functions which computes 
    .. math::
        A^{-1} (e^A - I)
        
    and importantly deals with potential numerical instability in the expression.
    """
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

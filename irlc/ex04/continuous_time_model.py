# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her23] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2023.
"""
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import tabulate

class ContiniousTimeSymbolicModel: 
    """
    The continious time symbolic model. See (Her23, Section 11.3) for a top-level description.

    The model represents the physical system we are simulating and can be considered a control-equivalent of the
    :class:`irlc.ex02.dp_model.DPModel`. The class must keep track of the following:

    .. math::
        \\frac{dx}{dt} = f(x, u, t)

    And the cost-function which is defined as an integral

    .. math::
        c_F(t_0, t_F, x(t_0), x(t_F)) + \int_{t_0}^{t_F} c(t, x, u) dt

    as well as constraints and boundary conditions on :math:`x`, :math:`u` and the initial conditions state :math:`x(t_0)`.
    this course, the cost function will always be quadratic, and can be accessed as ``model.cost``.

    If you want to implement your own model, the best approach is to start with an existing model and modify it for
    your needs. The overall idea is that you specify the dimensions of ``x`` and ``u`` by setting the boundary conditions in
    ``model.bounds`` (each bound is simply a list, and the dimension is the length of the list),
    and then define the dynamics by modifying the ``sym_f`` function. Note this function should accept symbolic
    expressions as input and return a symbolic expression. Bounds that you don't care about for your application can
    just be set to ``[np.inf, np.inf, ...]`` to signify no bound is applied.

    This class will then automatically convert the dynamics into a numpy function, and allow e.g. the discrete model to
    compute derivatives.
    """

    """
    This dictionary contains all simple bounds (constraints) applied to the model. A simple bound on e.g. the starting position `:math:`x_0`
    is of the form
    
    .. math::
        x_{0,lb} \leq x_0 \leq x_{0,ub}

    To signify a variable is not bounded, set the corresponding entry to :math:`\infty`. To access the bounds use:

    .. runblock:: pycon

        >>> from irlc.ex04.model_pendulum import ContiniousPendulumModel
        >>> model = ContiniousPendulumModel()
        >>> model.bounds['x0_low']
        >>> model.bounds['x0_high']
        >>> bounds.keys() # Get all bounds. 

    The bounds contained in the dictionary are:

        - ``"x0"`` - Bound on starting position :math:`x_0`
        - ``"xF"`` - Bound on terminal position :math:`x_F`
        - ``"x"`` - Bound on states :math:`x_0`
        - ``"u"`` - Bound on actions :math:`u_0`
        - ``"t0"`` - Bound on initial (starting) time :math:`t_0` (0 in all our examples)
        - ``"tF"`` - Bound on terminal (stopping) time :math:`t_F`            
    """
    bounds = None

    state_labels = None     # Labels (as lists) used for visualizations.
    action_labels = None    # Labels (as lists) used for visualizations.

    def __init__(self, cost, bounds=None): 
        """
        The cost must be an instance of :class:`irlc.ex04.cost_continuous.SymbolicQRCost`.
        Bounds is a dictionary but otherwise optional; the model should give it a default value.

        :param cost: A quadratic cost function
        :param dict bounds: A dictionary of boundary constraints.
        """
        self.cost = cost 

        # The model requires some bounds, otherwise it cannot guess the state/action dimensions.
        if bounds is None:
            raise Exception("No bounds specified. You must at least specify x_low, x_high and u_low, u_high")

        # Try to guess missing bounds.
        self.bounds = bounds
        self.bounds['t0_low'] = self.bounds.get('t0_low',0)
        self.bounds['t0_high'] = self.bounds.get('t0_high', 0)
        self.bounds['tF_low'] = self.bounds.get('tF_low', self.bounds['t0_high'])
        self.bounds['tF_high'] = self.bounds.get('tF_high', np.inf)
        for lh in ['low', 'high']:
            for v in ['0', 'F']:
                self.bounds[f'x{v}_{lh}'] = self.bounds.get(f'x{v}_{lh}', self.bounds[f'x_{lh}'])

        if self.state_labels is None:
            self.state_labels = [f'x{i}' for i in range(self.state_size)]
        if self.action_labels is None:
            self.action_labels = [f'u{i}' for i in range(self.action_size)]

        t = sym.symbols("t") 
        x = symv("x", self.state_size)
        u = symv("u", self.action_size)
        self.f = sym.lambdify((x, u, t), self.sym_f(x, u, t))  

    def sym_f(self, x, u, t=None): 
        """
        The symbolic (``sympy``) version of the dynamics :math:`f(x, u, t)`. This is the main place where you specify
        the dynamics when you build a new model. you should look at concrete implementations of models for specifics.

        :param x: A list of symbolic expressions ``['x0', 'x1', ..]`` corresponding to :math:`x`
        :param u: A list of symbolic expressions ``['u0', 'u1', ..]`` corresponding to :math:`u`
        :param t: A single symbolic expression corresponding to the time :math:`t` (seconds)
        :return: A list of symbolic expressions ``[f0, f1, ...]`` of the same length as ``x`` where each element is a coordinate of :math:`f`
        """
        raise NotImplementedError("Implement a function which return the environment dynamics f(x,u,t) as a sympy exression") 

    def simulate(self, x0, u_fun, t0, tF, N_steps=1000, method='rk4'):  
        """
        Used to simulate the effect of a policy on the model. By default, it uses
        Runge-Kutta 4 (RK4) with a fine discretization -- this is slow, but in nearly all cases exact. See (Her23, Algorithm 18) for more information.

        The input argument ``u_fun`` should be a function which returns a list or tuple with same dimension as
        ``model.action_space``, :math:`d`.

        :param x0: The initial state of the simulation. Must be a list of floats of same dimension as ``env.observation_space``, :math:`n`.
        :param u_fun: Can be either:
            - Either a policy function that can be called as ``u_fun(x, t)`` and returns an action ``u`` in the ``action_space``
            - A single action (i.e. a list of floats of same length as the action space). The model will be simulated with a constant action in this case.
        :param float t0: Starting time :math:`t_0`
        :param float tF: Stopping time :math:`t_F`; the model will be simulated for :math:`t_F - t_0` seconds
        :param int N_steps: Steps :math:`N` in the RK4 simulation
        :param str method: Simulation method. Either ``'rk4'`` (default) or ``'euler'``
        :return:
            - xs - A numpy ``ndarray`` of dimension :math:`(N+1)\\times n` containing the observations :math:`x`
            - us - A numpy ``ndarray`` of dimension :math:`(N+1)\\times d` containing the actions :math:`u`
            - ts - A numpy ``ndarray`` of dimension :math:`(N+1)` containing the corresponding times :math:`t` (seconds)
        """

        u_fun = ensure_policy(u_fun)
        tt = np.linspace(t0, tF, N_steps+1)   # Time grid t_k = tt[k] between t0 and tF.
        xs = [ np.asarray(x0) ]
        us = [ u_fun(x0, t0 )]
        for k in range(N_steps):
            Delta = tt[k+1] - tt[k]
            tn = tt[k]
            xn = xs[k]
            un = us[k]   # ensure the action u is a vector.
            unp = u_fun(xn, tn + Delta)
            if method == 'rk4':
                """ Implementation of RK4 here. See: (Her23, Algorithm 18) """
                k1 = np.asarray(self.f(xn, un, tn))
                k2 = np.asarray(self.f(xn + Delta * k1/2, u_fun(xn, tn+Delta/2), tn+Delta/2))
                k3 = np.asarray(self.f(xn + Delta * k2/2, u_fun(xn, tn+Delta/2), tn+Delta/2))
                k4 = np.asarray(self.f(xn + Delta * k3,   u_fun(xn, tn + Delta), tn+Delta))
                xnp = xn + 1/6 * Delta * (k1 + 2*k2 + 2*k3 + k4)
            elif method == 'euler':
                xnp = xn + Delta * np.asarray(self.f(xn, un, tn))
            else:
                raise Exception("Bad integration method", method)
            xs.append(xnp)
            us.append(unp)

        xs = np.stack(xs, axis=0)
        us = np.stack(us, axis=0)
        return xs, us, tt 

    @property
    def state_size(self):
        """
        This field represents the dimensionality of the state-vector :math:`n`. Use it as ``model.state_size``
        :return: Dimensionality of the state vector :math:`x`
        """
        return len(list(self.bounds['x_low']))

    @property
    def action_size(self):
        """
        This field represents the dimensionality of the action-vector :math:`d`. Use it as ``model.action_size``
        :return: Dimensionality of the action vector :math:`u`
        """
        return len(list(self.bounds['u_low']))

    def render(self, x, render_mode="human"):
        """
        Responsible for rendering the state. You don't have to worry about this function.

        :param x: State to render
        :param str render_mode: Rendering mode. Select ``"human"`` for a visualization.
        :return:  Either none or a ``ndarray`` for plotting.
        """
        raise NotImplementedError()

    def sym_h(self, x, u, t):
        """
        (Note: You will only need to use this function for the Brachistochrone problem).
        The function represent all dynamical path-constraints affecting the model as

        .. math::
            h(x, u, t) \leq 0

        The input variables ``x`` and ``u`` are (lists) of symbolic expressions, and the function should return a (list) of
        symbolic expressions representing all the constraints.

        :param x: Lists of symbolic variables representing the coordinates of the state
        :param u: Lists of symbolic variables representing the coordinates of the action
        :param t: The current time as a symbolic variable
        :return: A list of symbolic expressions ``[h1, h2, ... hm]`` so that :math:`h_j(x, u, t) \leq 0`
        """
        return []

    def close(self):
        pass

    def guess(self):
        def mfin(z):
            return [z_ if np.isfinite(z_) else 0 for z_ in z]
        xL = mfin(self.bounds['x0_low'])
        xU = mfin(self.bounds['xF_high'])
        tF = 10 if not np.isfinite(self.bounds['tF_high']) else self.bounds['tF_high']
        gs = {'t0': 0,
              'tF': tF,
              'x': [xL, xU],
              'u': [mfin(self.bounds['u_low']), mfin(self.bounds['u_high'])]}
        return gs

    def __str__(self):
        """
        Return a string representation of the model. This is a potentially helpful way to summarize the content of the
        model. You can use it as:

        .. runblock:: pycon

            >>> from irlc.ex04.model_pendulum import ContiniousPendulumModel
            >>> model = ContiniousPendulumModel()
            >>> print(model)

        :return: A string containing the details of the model.
        """
        split = "-"*20
        s = [f"{self.__class__}"] + ['='*50]
        s += ["Dynamics:", split]
        t = sym.symbols("t")
        x = symv("x", self.state_size)
        u = symv("u", self.action_size)
        s += [f"f({x}, {u}) = {str(self.sym_f(x, u, t))}"]
        s += ["Cost:", split, str(self.cost)]
        s += ["Bounds:", split]
        dd = defaultdict(list)
        for k in self.bounds:
            v, ex = k.split("_")
            if ex == 'low':
                dd['low'].append(self.bounds[k])
                dd['variable'].append("<= " + v + " <=")
            else:
                dd['high'].append(self.bounds[k])
        s += [tabulate.tabulate(dd, headers='keys')]
        return "\n".join(s)


def symv(s, n):
    """
    Returns a vector of symbolic functions. For instance if s='x' and n=3 then it will return
    [x0,x1,x2]
    where x0,..,x2 are symbolic variables.
    """
    return sym.symbols(" ".join(["%s%i," % (s, i) for i in range(n)]))

def ensure_policy(u):
    """
    Ensure u corresponds to a policy function with input arguments u(x, t)
    """
    if callable(u):
        return lambda x, t: np.asarray(u(x,t)).reshape((-1,))
    else:
        return lambda x, t: np.asarray(u).reshape((-1,))

def plot_trajectory(x_res, tt, lt='k-', ax=None, labels=None, legend=None):
    M = x_res.shape[1]
    if labels is None:
        labels = [f"x_{i}" for i in range(M)]

    if ax is None:
        if M == 2:
            a = 234
        if M == 3:
            r = 1
            c = 3
        else:
            r = 2 if M > 1 else 1
            c = (M + 1) // 2

        H = 2*r if r > 1 else 3
        W = 6*c
        # if M == 2:
        #     W = 12
        f, ax = plt.subplots(r,c, figsize=(W,H))
        if M == 1:
            ax = np.asarray([ax])
        print(M,r,c)

    for i in range(M):
        if len(ax) <= i:
            print("issue!")

        a = ax.flat[i]
        a.plot(tt, x_res[:, i], lt, label=legend)

        a.set_xlabel("Time/seconds")
        a.set_ylabel(labels[i])
        # a.set_title(labels[i])
        a.grid(True)
        if legend is not None and i == 0:
            a.legend()
        # if i == M:
    plt.tight_layout()
    return ax

def make_space_above(axes, topmargin=1.0):
    """ increase figure size to make topmargin (in inches) space for
        titles, without changing the axes sizes"""
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)

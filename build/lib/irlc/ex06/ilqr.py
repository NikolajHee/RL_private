# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
This implements two methods: The basic ILQR method, described in (Her23, Algorithm 24), and the linesearch-based method
described in (Her23, Algorithm 25).

If you are interested, you can consult (TET12) (which contains generalization to DDP) and (Har20, Alg 1).
References:
  [Her23] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2023.
  [TET12] Yuval Tassa, Tom Erez, and Emanuel Todorov. Synthesis and stabilization of complex behaviors through online trajectory optimization. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 4906–4913. IEEE, 2012. (See tassa2012.pdf).
  [Har20] James Harrison. Optimal and learning-based control combined course notes. (See AA203combined.pdf), 2020.
"""
import warnings
import numpy as np
from irlc.ex06.dlqr import LQR

def ilqr_basic(model, N, x0, us_init=None, n_iterations=500,verbose=True):
    '''
    Basic ilqr. I.e. (Her23, Algorithm 24). Our notation (x_bar, etc.) will be consistent with the lecture slides
    '''
    mu, alpha = 1, 1 # Hyperparameters. For now, just let them have defaults and don't change them
    # Create a random initial state-sequence
    n, m = model.state_size, model.action_size
    u_bar = [np.random.uniform(-1, 1,(model.action_size,)) for _ in range(N)] if us_init is None else us_init
    x_bar = [x0] + [np.zeros(n, )] * (N)
    """
    Initialize nominal trajectory xs, us using us and x0 (i.e. simulate system from x0 using action sequence us). 
    The simplest way to do this is to call forward_pass with all-zero sequence of control vector/matrix l, L.
    """
    # TODO: 2 lines missing.
    raise NotImplementedError("Initialize x_bar, u_bar here")
    J_hist = []
    for i in range(n_iterations):
        """
        Compute derivatives around trajectory and cost estimate J of trajectory. To do so, use the get_derivatives
        function. Remember the functions will return lists of derivatives.
        """
        # TODO: 2 lines missing.
        raise NotImplementedError("Compute J and derivatives A_k = f_x, B_k = f_u, ....")
        """  Backward pass: Obtain feedback law matrices l, L using the backward_pass function.
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Compute L, l = .... here")
        """ Forward pass: Given L, l matrices computed above, simulate new (optimal) action sequence. 
        In the lecture slides, this is similar to how we compute u^*_k and x_k
        Once they are computed, iterate the iLQR algorithm by setting x_bar, u_bar equal to these values
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Compute x_bar, u_bar = ...")
        if verbose:
            print(f"{i}> J={J:4g}, change in cost since last iteration {0 if i == 0 else J-J_hist[-1]:4g}")
        J_hist.append(J)
    return x_bar, u_bar, J_hist, L, l

def ilqr_linesearch(model, N, x0, n_iterations, us_init=None, tol=1e-6,verbose=True):
    """
    For linesearch implement method described in (Her23, Algorithm 25) (we will use regular iLQR, not DDP!)
    """
    # The range of alpha-values to try out in the linesearch
    # plus parameters relevant for regularization scheduling.
    alphas = 1.1 ** (-np.arange(10) ** 2)  # alphas = [1, 1.1^{-2}, ...]
    mu_min = 1e-6
    mu_max = 1e10
    Delta_0 = 2
    mu = 1.0
    Delta = Delta_0

    n, m = model.state_size, model.action_size
    u_bar = [np.random.uniform(-1, 1, (model.action_size,)) for _ in range(N)] if us_init is None else us_init
    x_bar = [x0] + [np.zeros(n, )] * (N)
    # Initialize nominal trajectory xs, us (same as in basic linesearch)
    # TODO: 2 lines missing.
    raise NotImplementedError("Copy-paste code from previous solution")
    J_hist = []

    converged = False
    for i in range(n_iterations):
        alpha_was_accepted = False
        """ Step 1: Compute derivatives around trajectory and cost estimate of trajectory.
        (copy-paste from basic implementation). In our implementation, J_bar = J_{u^star}(x_0) """
        # TODO: 2 lines missing.
        raise NotImplementedError("Obtain derivatives f_x, f_u, ... as well as cost of trajectory J_bar = ...")
        try:
            """
            Step 2: Backward pass to obtain control law (l, L). Same as before so more copy-paste
            """
            # TODO: 1 lines missing.
            raise NotImplementedError("Obtain l, L = ... in backward pass")
            """
            Step 3: Forward pass and alpha scheduling.
            Decrease alpha and check condition |J^new < J'|. Apply the regularization scheduling as needed. """
            for alpha in alphas:
                x_hat, u_hat = forward_pass(model, x_bar, u_bar, L=L, l=l, alpha=alpha) # Simulate trajectory using this alpha
                # TODO: 1 lines missing.
                raise NotImplementedError("Compute J_new = ... as the cost of trajectory x_hat, u_hat")

                if J_new < J_prime:
                    """ Linesearch proposed trajectory accepted! Set current trajectory equal to x_hat, u_hat. """
                    if np.abs((J_prime - J_new) / J_prime) < tol:
                        converged = True  # Method does not seem to decrease J; converged. Break and return.

                    J_prime = J_new
                    x_bar, u_bar = x_hat, u_hat
                    '''
                    The update was accepted and you should change the regularization term mu, 
                     and the related scheduling term Delta.                   
                    '''
                    # TODO: 1 lines missing.
                    raise NotImplementedError("Delta, mu = ...")
                    alpha_was_accepted = True # accept this alpha
                    break
        except np.linalg.LinAlgError as e:
            # Matrix in dlqr was not positive-definite and this diverged
            warnings.warn(str(e))

        if not alpha_was_accepted:
            ''' No alphas were accepted, which is not too hot. Regularization should change
            '''
            # TODO: 1 lines missing.
            raise NotImplementedError("Delta, mu = ...")

            if mu_max and mu >= mu_max:
                raise Exception("Exceeded max regularization term; we are stuffed.")

        dJ = 0 if i == 0 else J_prime-J_hist[-1]
        info = "converged" if converged else ("accepted" if alpha_was_accepted else "failed")
        if verbose:
            print(f"{i}> J={J_prime:4g}, decrease in cost {dJ:4g} ({info}).\nx[N]={x_bar[-1].round(2)}")
        J_hist.append(J_prime)
        if converged:
            break
    return x_bar, u_bar, J_hist, L, l

def backward_pass(A, B, c_x, c_u, c_xx, c_ux, c_uu, _mu=1):
    """
    Get L,l feedback law given linearization around nominal trajectory
    To do so, simply call LQR with appropriate inputs (i.e. the derivative terms).

    Remember the terminal costs terms, i.e. that
         c = [c_{0}, ..., c_{N-1}, c_{N}].
         c_x = [c_{xx,0}, ..., c_{x,N-1}, c_{x,N}].
         c_xx = [c_{xx,0}, ..., c_{xx,N-1}, c_{xx,N}].
    """
    # TODO: 5 lines missing.
    raise NotImplementedError("Insert your solution and remove this error.")
    (L, l), (V, v, vc) = LQR(A=A, B=B, R=R, Q=Q, QN=QN, H=H, q=q, qN=qN, r=r, mu=_mu)
    return L,l

def compute_J(model, xs, us):
    """
    Helper function which computes the cost of the trajectory. 
    
    Input: 
        xs: States (N+1) x [(state_size)]
        us: Actions N x [(state_size)]
        
    Returns:
        Trajectory's total cost.
    """
    N = len(us)
    JN = model.cost.cN(xs[-1])
    return sum(map(lambda args:  model.cost.c(*args), zip(xs[:-1], us, range(N)))) + JN

def get_derivatives(model, x_bar, u_bar):
    """
    Compute derivatives for system dynamics around the given trajectory. should be handled using
    model.f and model.cost.c, model.cost.cN.

    The return type has the same meaning as in the notes. I.e. the function should return lists so that:

    A = [A_0, A_1, ..., A_{N-1}]
    B = [B_0, B_1, ..., B_{N-1}]

    Each term is computed by taking the derivative of the dynamics (as in the notes)), i.e. A_k = df/dx(x^bar_k)

    Meanwhile the terms c, c_x, ... has the meaning described in (Her23, Subequation 15.8), i.e. the derivatives of the c (cost) terms.
    These derivatives will be returned as lists of matrices/vectors, i.e. one for each k-value. Note that in particular
    c will be a N+1 list of the cost terms, such that J = sum(c) is the total cost of the trajectory:

    c = [c_0, ..., c_N]
    c_x = [c_{x,0}, ..., c_{x,N}]
    c_xx = [c_{xx,0}, ..., c_{xx,N}]

    c_u = [c_{u,0}, ..., c_{u,N-1}]
    c_ux = [c_{ux,0}, ..., c_{ux,N-1}]
    c_uu = [c_{uu,0}, ..., c_{uu,N-1}]

    """
    N = len(u_bar)
    """ Compute A_k, B_k (lists of matrices of length N) as the jacobians of the dynamics. To do so, 
    recall from the online documentation that: 
        x, f_x, f_u = model.f(x, u, k, compute_jacobian=True)
    """
    A = [None]*N
    B = [None]*N
    c = [None] * (N+1)
    c_x = [None] * (N + 1)
    c_xx = [None] * (N + 1)

    c_u = [None] * (N+1)
    c_ux = [None] * (N + 1)
    c_uu = [None] * (N + 1)
    # Now update each entry correctly (i.e., ensure there are no None elements left).
    # TODO: 2 lines missing.
    raise NotImplementedError("Insert your solution and remove this error.")
    """ Compute derivatives of the cost function. For terms not including u these should be of length N+1 
    (because of gN!), for the other lists of length N
    recall model.cost.c has output:
        c[i], c_x[i], c_u[i], c_xx[i], c_ux[i], c_uu[i] = model.cost.c(x, u, i, compute_gradients=True)
    """
    # TODO: 2 lines missing.
    raise NotImplementedError("Insert your solution and remove this error.")
    # Concatenate the derivatives associated with the last time point N.
    cN, c_xN, c_xxN = model.cost.cN(x_bar[N], compute_gradients=True)
    # TODO: 3 lines missing.
    raise NotImplementedError("Update c, c_x and c_xx with the terminal terms.")
    return A, B, c, c_x, c_u, c_xx, c_ux, c_uu

def forward_pass(model, x_bar, u_bar, L, l, alpha=1.0):
    """Applies the controls for a given trajectory.

    Args:
        x_bar: Nominal state path [N+1, state_size].
        u_bar: Nominal control path [N, action_size].
        l: Feedforward gains [N, action_size].
        L: Feedback gains [N, action_size, state_size].
        alpha: Line search coefficient.

    Returns:
        Tuple of
            x: state path [N+1, state_size] simulated by the system
            us: control path [N, action_size] new control path
    """
    N = len(u_bar)
    x = [None] * (N+1)
    u_star = [None] * N
    x[0] = x_bar[0].copy()

    for i in range(N):
        """ Compute using (Her23, eq. (15.16))
        u_{i} = ...
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("u_star[i] = ....")
        """ Remember to compute 
        x_{i+1} = f_k(x_i, u_i^*)        
        here:
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("x[i+1] = ...")
    return x, u_star

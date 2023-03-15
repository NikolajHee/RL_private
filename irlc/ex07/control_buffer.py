# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her23] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2023.
"""
import numpy as np
import faiss # Facebook AI software which has a super fast nearest-neighbour implementation.

class Buffer:  
    """
    A class which implements a simple replay buffer to store and retrieve control observations as used in (Her23, Algorithm 27).
    It assumes we observe transitions :math:`x_k, u_k, x_{k+1}` from a discrete system:

    .. math::
        x_{k+1} = f(x_k, u_k)

    These can then be stored in the buffer using the :func:`~irlc.ex07.control_buffer.Buffer.push` function.

    Once the buffer contains observations, we can either get all observations in the buffer, or only those that are closest
    to a state/action pair corresponding to a system state we wish to approximate. The later will be useful for local linear regression.

    The following example shows how we can push a number of transitions into the buffer and then get those closest to
    a state/action pair :math:`x, u` of interest:

    .. runblock:: pycon

        >>> from irlc.ex04.model_pendulum import GymSinCosPendulumEnvironment
        >>> from irlc.ex07.control_buffer import Buffer
        >>> env = GymSinCosPendulumEnvironment()
        >>> buffer = Buffer()
        >>> x, _ = env.reset()
        >>> for _ in range(100):
        ...     u = env.action_space.sample()
        ...     xp, _, _, _, _ = env.step(u)
        ...     buffer.push(x, u, xp) # Add to the buffer
        ...     x = xp
        >>>
        >>> print("Size of buffer is", len(buffer))
        >>> X, U, XP = buffer.get_data() # Get all data
        >>> X.shape # Note observations are stacked vertically
        >>> X, U, XP = buffer.get_closest_observations(x, u, N=50) # Get the 50 closest observations to (x,u).
        >>> U.shape # again, stacked vertically

    """
    def __init__(self):
        self.x = []
        self.u = []
        self.xp = []

    def push(self, x, u, xp):  
        """
        Add the transition :math:`(x_k, u_k, x_{k+1})` to the buffer.

        :param x: A state vector
        :param u: an action vector
        :param xp: the state we immediately transitioned to
        :return: ``None``
        """
        if len(self.x) == 0: 
            self.index = faiss.index_factory(x.size+ u.size, "Flat")

        self.x.append(x)
        self.u.append(u)
        self.xp.append(xp)
        NN_WITH_U = True
        Z = np.concatenate([x, u]) if NN_WITH_U else x
        self.index.add(Z[np.newaxis, :])

    def __len__(self):
        return len(self.x)

    def get_data(self): 
        """
        Return all :math:`N` transitions contained in the buffer. The output will be stacked vertically so that:

        .. math::
            X = \\begin{bmatrix} x_1^\\top  \\\\  x_2^\\top \\\\ \\cdots \\\\ x_N^\\top \\end{bmatrix}

        is a :math:`N \\times n` matrix of states.

        :return:
            - X - A :math:`N \\times n` matrix of states, each row corresponding to a :math:`x_k`
            - U - A :math:`N \\times d` matrix of actions, each row corresponding to a :math:`x_k`
            - XP - A :math:`N \\times n` matrix of immediately following states, each row corresponding to a :math:`x_{k+1}`
        """
        X = np.asarray(self.x)  # train new LQR 
        XP = np.asarray(self.xp)
        U = np.asarray(self.u)
        return X, U, XP 

    def get_closest_observations(self, x, u, N=50): 
        """
        Return the :math:`n` transitions contained in the buffer which are the closest to the input point :math:`x, u` in terms of Euclidian distance.
        That is, the observations :math:`x', u'` in the buffer which minimize the Euclidian distance:

        .. math::
            \\left\\| \\begin{bmatrix} x \\\\ u \\end{bmatrix} - \\begin{bmatrix} x' \\\\ u' \\end{bmatrix} \\right\\|  

        These will be returned as matrices which are stacked vertically. I.e. the first return argument will have the form:

        .. math::
            X = \\begin{bmatrix} x_1^\\top  \\\\  x_2^\\top \\\\ \\cdots \\\\ x_N^\\top \\end{bmatrix}

        and will be a :math:`N \\times n` matrix of states.


        :param x: A state vector the output transitions should be close to
        :param u: An action vector the output transitions should be close to
        :param N: The number of closest transitions to return
        :return:
            - X - A :math:`N \\times n` matrix of states, each column corresponding to a :math:`x_k`
            - U - A :math:`N \\times d` matrix of actions, each column corresponding to a :math:`x_k`
            - XP - A :math:`N \\times n` matrix of immediately following states, each column corresponding to a :math:`x_{k+1}`
        """ 
        if len(self) < N:
            return self.get_data()
        X, U, XP = self.get_data()
        TT = 1
        NN_WITH_U = True

        def nnfaiss(Z, n, z):
            distances, neighbors = self.index.search(z.reshape(1, -1).astype(np.float32), n)
            return distances, neighbors

        for _ in range(TT):
            Z = None
            z = (np.concatenate([x,u],axis=0) if NN_WITH_U else x).reshape( (1,-1))
            distances, indices = nnfaiss(Z, N, z)

            # distances, indices = nn.kneighbors(z)
            indices = indices.squeeze()
            xx, uu,xxp = X[indices], U[indices], XP[indices]
        return xx, uu, xxp 


if __name__ == "__main__":

    from irlc.ex04.model_pendulum import GymSinCosPendulumEnvironment 
    env = GymSinCosPendulumEnvironment()
    buffer = Buffer()
    x, _ = env.reset()
    for _ in range(100):
        u = env.action_space.sample()
        xp, _, _, _, _ = env.step(u)
        buffer.push(x, u, xp)  # Add to the buffer
        x = xp

    X, U, XP = buffer.get_data()  # Get all data
    X, U, XP = buffer.get_closest_observations(x, u, N=50)  # 50 closest to (x,u) 

    from irlc.ex07.regression import solve_linear_problem_simple 
    A, B, d = solve_linear_problem_simple(XP, X, U) 

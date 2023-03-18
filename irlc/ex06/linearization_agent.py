# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her23] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2023.
"""
from irlc.ex06.dlqr import LQR
from irlc import Agent
# from irlc import VideoMonitor
from irlc.ex04.model_cartpole import GymSinCosCartpoleEnvironment
from irlc import train, savepdf
import matplotlib.pyplot as plt
import numpy as np

class LinearizationAgent(Agent):
    """ Implement the simple linearization procedure described in (Her23, Algorithm 23) which expands around a single fixed point. """
    def __init__(self, env, model, xbar=None, ubar=None):
        self.model = model
        N = 50  # Plan on this horizon. The control matrices will converge fairly quickly.
        """ Define A, B, d as the list of A/B matrices here. I.e. x[t+1] = A x[t] + B u[t] + d.
        You should use the function model.f to do this, which has build-in functionality to compute Jacobians which will be equal to A, B.
        It is important that you linearize around xbar, ubar. See (Her23, Section 15.1) for further details. """
        # TODO: 2 lines missing.
        xp, A, B = model.f(xbar, ubar, k=0, compute_jacobian=True) 
        d = xp - A @ xbar - B @ ubar 
        Q, q, R = self.model.cost.Q, self.model.cost.q, self.model.cost.R
        """ Define self.L, self.l here as the (lists of) control matrices. """
        # TODO: 1 lines missing.
        (self.L, self.l), (V, v, vc) = LQR(A=[A]*N, B=[B]*N, d=[d]*N, Q=[Q]*N, q=[q]*N, R=[self.model.cost.R]*N) 
        super().__init__(env)

    def pi(self, x, k, info=None):
        """
        Compute the action here using u_k = L_0 x_k + l_0. The control matrix/vector L_0 can be found as the output from LQR, i.e.
        L_0 = L[0] and l_0 = l[0].

        The reason we use L_0, l_0 (and not L_k, l_k) is because the LQR problem itself is an approximation of the true dynamics
        and this controller will be able to balance the pendulum for an infinite amount of time.
        """
        # TODO: 1 lines missing.
        u = self.L[0] @ x + self.l[0] 
        return u


def get_offbalance_cart(waiting_steps=30, sleep_time=0.1):
    env = GymSinCosCartpoleEnvironment(Tmax=3, render_mode='human')
    env.reset()
    import time
    time.sleep(sleep_time)
    env.state = env.discrete_model.x_upright
    env.state[-1] = 0.01 # a bit of angular speed.
    for _ in range(waiting_steps):  # Simulate the environment for 30 steps to get things out of balance.
        env.step(1)
        time.sleep(sleep_time)
    return env


if __name__ == "__main__":
    np.random.seed(42) # I don't think these results are seed-dependent but let's make sure.
    from irlc import plot_trajectory
    env = get_offbalance_cart(4) # Simulate for 4 seconds to get the cart off-balance. Same idea as PID control.
    agent = LinearizationAgent(env, model=env.discrete_model, xbar=env.discrete_model.x_upright, ubar=env.action_space.sample()*0)
    _, trajectories = train(env, agent, num_episodes=1, return_trajectory=True, reset=False)  # Note reset=False to maintain initial conditions.
    plot_trajectory(trajectories[0], env, xkeys=[0,2, 3], ukeys=[0])
    env.close()
    savepdf("linearization_cartpole")
    plt.show()

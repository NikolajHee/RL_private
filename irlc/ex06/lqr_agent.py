# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex06.dlqr import LQR
from irlc import Agent

class DiscreteLQRAgent(Agent):
    def __init__(self, env, model):
        self.model = model
        N = int(env.Tmax / env.dt) # Obtain the planning horizon
        """ Define A, B as the list of A/B matrices here. I.e. x[t+1] = A x[t] + B x[t] + d.
        You should use the function model.f to do this, which has build-in functionality to compute Jacobians which will be equal to A, B """
        # TODO: 1 lines missing.
        d, A, B = model.f(env.observation_space.sample()*0, env.action_space.sample()*0, k=0, compute_jacobian=True) 
        #raise NotImplementedError("Insert your solution and remove this error.")
        Q, q, R = self.model.cost.Q, self.model.cost.q, self.model.cost.R
        """ Define self.L, self.l here as the (lists of) control matrices. """
        # TODO: 1 lines missing.
        (self.L, self.l), (V, v, vc) = LQR(A=[A]*N, B=[B]*N, d=[d]*N, Q=[Q]*N, q=[q]*N, R=[self.model.cost.R]*N) 
        self.dt = env.dt
        super().__init__(env)

    def pi(self,x, k, info=None):
        """
        Compute the action here using u = L_k x + l_k.
        You should use self.L, self.l to get the control matrices (i.e. L_k = self.L[k] ),
        """
        # k = int(t / self.env.dt)
        # TODO: 1 lines missing.
        u = self.L[k] @ x + self.l[k]  
        return u

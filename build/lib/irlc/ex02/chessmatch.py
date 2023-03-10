# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her23] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2023.
"""
import numpy as np
from irlc.ex02.dp_model import DPModel
from irlc.ex02.dp import DP_stochastic


class ChessMatch(DPModel):
    """
    See (Her23, Subsection 6.2.2) for details on this problem.

    Note that timid play will be denoted by the action u=0, bold play by u=1. The state is represented as an integer
    which tracks the score, i.e. x=1 means we are ahead by one point and x=0 means the match is tied.
    """
    def __init__(self, N, pw, pd):
        self.pw = pw
        self.pd = pd
        super(ChessMatch, self).__init__(N)

    def A(self, x, k): 
        # TODO: 1 lines missing.
        raise NotImplementedError("Return action space (hint: there are two actions; timid and bold play)")

    def S(self, k): 
        """
        State space is {-k, ..., k} (maximal loss to maximal win)
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def g(self, x, u, w, k):  
        # TODO: 1 lines missing.
        raise NotImplementedError("Note that g_k(x, u, w) = 0")

    def f(self, x, u, w, k): 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def Pw(self, x, u, k):
        """
        Should return win/loss probabilities depending on u.
        In either case, return a dict of the form: {w1: p(w1), w2: p(w2), ...}
        Note w is whether we win, draw or loose (+1, 0, -1) and the probabilities
        are given in the problem statement (see self.pd, self.pw)
        """
        if u == 0:  # timid play
            return {-1: 1 - self.pd,  # loss.
                    0: self.pd}       # draw
        else: # Bold play. return something quite similar to the timid-case but with the right probabilities.
            # TODO: 1 lines missing.
            raise NotImplementedError("compute and return policy corresponding to bold play.")

    def gN(self, x):
        """
        Return cost (-reward) dependin on final match score. Should for instance return
        -1 (reward=1) in case we win (x>0) and (-'drawing chance') in case of a drawn match.
        """
        if x > 0:
            return -1
        # TODO: 4 lines missing.
        raise NotImplementedError("Compute and return terminal costs in the two other cases.")

def policy_rollout(model, pi, x0):
    x = x0
    J = 0
    for k in range(model.N):
        u = pi(x, k)
        w = model.w_rnd(x, u, k)
        J += model.g(x, u, w, k)
        x = model.f(x, u, w, k)
    J += model.gN(x)
    return J

def pi_smart(x, k): 
    """ smart policy: archives optimal match-win probability """
    # TODO: 1 lines missing.
    raise NotImplementedError("Implement function body")

if __name__ == '__main__':
    """
    Chess match problem, see (Her23, Subsection 6.2.2) for details on this problem.
    
    Since the problem is formulated as reward, we multiply the reward by -1 to get a cost. 
    """
    N = 2 
    pw = 0.45
    pd = 0.8
    cm = ChessMatch(N, pw=pw, pd=pd)

    T = 50000  # MC evaluation of policy
    J = np.mean([policy_rollout(cm, pi_smart, x0=0) for _ in range(T)])
    pW = pw * (pw + (pw + pd) * (1 - pw))
    print(f"Expected reward (-cost) when starting from a match score of 0: {-J} (true value {pW})")
    """
    Train and evaluate the chess match.
    """
    J, pi = DP_stochastic(cm) 
    print(f"Expected reward (-cost) when starting from a match score of 0: {-J[0][0]} (true value {pW})")
    print(f"value of J:")
    for k,Jk in enumerate(J):
        for x,Jx in enumerate(Jk):
            print(f"J_{k}({x}) = {Jx}")
        print("---")
    print(f"Policy at k=0: ", pi[0])
    print(f"Policy at k=1: ", pi[1])

    def dp_pi(x,k): 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    J_dp_pi = np.mean([policy_rollout(cm, dp_pi, x0=0) for _ in range(T)])
    print(f"Expected reward (-cost) when starting from a match score of 0: {-J_dp_pi} (true value {pW})") 

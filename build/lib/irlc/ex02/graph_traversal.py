# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her23] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2023.
"""
import numpy as np
from irlc.ex02.dp_model import DPModel

"""
Graph of shortest path problem of (Her23, Subsection 5.1.1)
"""
G222 = {(1, 2): 6,  (1, 3): 5, (1, 4): 2, (1, 5): 2,  
        (2, 3): .5, (2, 4): 5, (2, 5): 7,
        (3, 4): 1,  (3, 5): 5, (4, 5): 3}  

def symG(G):
    """ make a graph symmetric. I.e. if it contains edge (a,b) with cost z add edge (b,a) with cost c """
    G.update({(b, a): l for (a, b), l in G.items()})
symG(G222)

class SmallGraphDP(DPModel):
    """ Implement the small-graph example in (Her23, Subsection 5.1.1). t is the terminal node. """
    def __init__(self, t, G=None):  
        self.G = G.copy() if G is not None else G222.copy()  
        self.G = self.G.copy()  # Copy G. This is good style since G is passed by reference & modified in place.
        self.G[(t,t)] = 0  # make target vertex absorbing  
        self.t = t         # target vertex in graph
        self.nodes = {i for k in self.G for i in k} # set of all nodes
        super(SmallGraphDP, self).__init__(N=len(self.nodes)-1)  

    def f(self, x, u, w, k):
        if (x,u) in self.G:  
            # TODO: 1 lines missing.
            return u
            #raise NotImplementedError("Implement function body")
        else:
            raise Exception("Nodes are not connected")

    def g(self, x, u, w, k): 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def gN(self, x):  
        # TODO: 1 lines missing.
        return 0 if x == self.t else np.inf
        raise NotImplementedError("Implement function body")

    def S(self, k):   
        return self.nodes

    def A(self, x, k):
        return {j for (i,j) in self.G if i == x} 

def pi_silly(x, k): 
    if x == 1:
        return 2
    else:
        return 1 

def pi_inc(x, k): 
    # TODO: 1 lines missing.
    return x+1
    raise NotImplementedError("Implement function body")

def pi_smart(x,k): 
    # TODO: 1 lines missing.
    return 5
    raise NotImplementedError("Implement function body")

def policy_rollout(model, pi, x0):
    """
    Given an environment and policy, should compute one rollout of the policy and compute
    cost of the obtained states and actions. In the deterministic case this corresponds to

    J_pi(x_0)

    in the stochastic case this would be an estimate of the above quantity.

    Note I am passing a policy 'pi' to this function. The policy is in this case itself a function, that is,
    you can write code such as

    > u = pi(x,k)

    in the body below.
    """
    J = 0
    x = x0
    trajectory = [x0]
    # Done with the initialization. Now we will do a rollout.
    ## TODO: Oy veh, the following 7 lines below have been permuted. Uncomment, rearrange to the correct order and remove the error.
    #-------------------------------------------------------------------------------------------------------------------------------
    for k in range(model.N):
        u = pi(x, k) # Generate the action u = ... here using the policy
        w = model.w_rnd(x, u, k) # This is required; just pass them to the transition function
        J += model.g(x, u, w, k)  # Add cost term g_k to the cost of the episode
        x = model.f(x, u, w, k)  # Update J and generate the next value of x.
        trajectory.append(x) # update the trajectory
    J += model.gN(x) # Add last cost term env.gN(x) to J.
    #raise NotImplementedError("Remove this exception after the above lines have been uncommented and rearranged.")
    return J, trajectory


def main():
    t = 5  # target node
    model = SmallGraphDP(t=t)
    x0 = 1  # starting node
    print("Cost of pi_silly", policy_rollout(model, pi_silly, x0)[0]) 
    print("Cost of pi_inc", policy_rollout(model, pi_inc, x0)[0])
    print("Cost of pi_smart", policy_rollout(model, pi_smart, x0)[0])  

if __name__ == '__main__':
    main()

# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from collections import defaultdict
from irlc import train
from irlc.ex02.dp_model import DPModel
from irlc.ex02.dp import DP_stochastic
from irlc.ex02.dp_agent import DynamicalProgrammingAgent
from irlc.pacman.pacman_environment import PacmanEnvironment
import numpy as np

east = """ 
%%%%%%%%
% P   .%
%%%%%%%% """ 

east2 = """
%%%%%%%%
%    P.%
%%%%%%%% """

SS2tiny = """
%%%%%%
%.P  %
% GG.%
%%%%%%
"""

SS0tiny = """
%%%%%%
%.P  %
%   .%
%%%%%%
"""

SS1tiny = """
%%%%%%
%.P  %
%  G.%
%%%%%%
"""

datadiscs = """
%%%%%%%
%    .%
%.P%% %
%.   .%
%%%%%%%
"""

# TODO: 30 lines missing
class PacManNoGhost(DPModel):
    def __init__(self, N, x0):
        super().__init__(N)
        # save all the possible states [S0,...,SN]
        self.S_save = get_future_states(x0, N)

    def f(self, x, u, w, k):
        # one deterministic move
        return list(p_next(x,u))[0]
    
    def g(self, x, u, w, k):
        # cost that increase with time until we win
        return 0 if (x.isWin()) else k
    
    def gN(self, x):
        # reward if we win
        return -1 if x.isWin() else 0
    
    def S(self, k):
        # index the Sk state-space
        return self.S_save[k]
    
    def A(self, x, k):
        # action space.
        return x.A()
    
class PacManWithGhost(DPModel):
    def __init__(self, N, x0):
        super().__init__(N)
        # save all the possible states [S0,...,SN]
        self.S_save = get_future_states(x0, N)

    def f(self, x, u, w, k):
        # Stochastic.
        dict_ = p_next(x,u)
        states = list(dict_.keys())
        return states[w]
    
    def g(self, x, u, w, k):
        # we dont care how long it takes anymore:
        return 0
    
    def gN(self, x):
        # reward if we win
        return -1 if x.isWin() else 0

    def S(self, k):
        # index the Sk state-space
        return self.S_save[k]

    def A(self, x, k):
        # action space.
        return x.A()

    def Pw(self, x, u, k):
        # A random noise calculated in the p_next function
        dict_ = p_next(x,u)
        return {i: p for (i,(_, p)) in enumerate(dict_.items())}


def p_next(x, u): 
    """ Given the agent is in GameState x and takes action u, the game will transition to a new state xp.
    The state xp will be random when there are ghosts. This function should return a dictionary of the form

    {..., xp: p, ...}

    of all possible next states xp and their probability -- you need to compute this probability.

    Hints:
        * In the above, xp should be a GameState, and p will be a float. These are generated using the functions in the GameState x.
        * Start simple (zero ghosts). Then make it work with one ghosts, and then finally with any number of ghosts.
        * Remember the ghosts move at random. I.e. if a ghost has 3 available actions, it will choose one with probability 1/3
        * The slightly tricky part is that when there are multiple ghosts, different actions by the individual ghosts may lead to the same final state
        * Check the probabilities sum to 1. This will be your main way of debugging your code and catching issues relating to the previous point.
    """
    # TODO: 8 lines missing.
    G = [{} for _ in range(x.players())]
    # the pacman makes one deterministic move
    x0 = x.f(u)
    G[0][x0] = 1
    # loop over every ghost (if no ghost -> no iteration)
    for i in range(x.players() - 1):
        # compute the possible states
        t1 = [xp.f(u) for (xp,p) in G[i].items() for u in xp.A()]
        # compute the probability (independence rule so we multiply)
        t2 = [p * 1/len(xp.A()) for (xp,p) in G[i].items() for u in xp.A()]
        # collect information in dictionary
        G[i+1] = {a: b for (a,b) in zip(t1, t2)}
    # sanity check
    assert abs(sum(G[-1].values()) - 1) < 1e-12
    # return result
    return G[-1]




def go_east(map, tmax = 1e4): 
    """ Given a map-string map (see examples in the top of this file) that can be solved by only going east, this will return
    a list of states Pacman will traverse. The list it returns should therefore be of the form:

    [s0, s1, s2, ..., sn]

    where each sk is a GameState object, the first element s0 is the start-configuration (corresponding to that in the Map),
    and the last configuration sn is a won GameState obtained by going east.

    Note this function should work independently of the number of required east-actions.

    Hints:
        * Use the GymPacmanEnvironment class. The report description will contain information about how to set it up, as will pacman_demo.py
        * Use this environment to get the first GameState, then use the recommended functions to go east
    """
    # TODO: 5 lines missing.
    x, _ = PacmanEnvironment(layout_str=east).reset()
    states = [x]
    t = 0
    # proceed east until we can't anymore
    while ("East" in x.A()) and t < tmax: # t is a max-iter
        x = x.f('East')
        states.append(x)
        t += 1
    return states

def get_future_states(x, N): 
    # TODO: 4 lines missing.
    state_spaces = [set([x])] # S0
    for i in range(N):
        # yes a list comprehension with 3 for loops ;)
        states = [u_new for k in state_spaces[i] for u in k.A() for u_new in list(p_next(k,u).keys())]
        # we append it as a set to avoid duplicates
        state_spaces.append(set(states))
    return state_spaces


def win_probability(map, N=10): 
    """ Assuming you get a reward of -1 on wining (and otherwise zero), the win probability is -J_pi(x_0). """
    # TODO: 5 lines missing.
    env = PacmanEnvironment(layout_str=map)
    initial_x, _ = env.reset()
    agent = DynamicalProgrammingAgent(env, model=PacManWithGhost(N, initial_x))
    win_probability = -agent.J[0][initial_x]
    return win_probability

def shortest_path(map, N=10): 
    """ If each move has a cost of 1, the shortest path is the path with the lowest cost.
    The actions should be the list of actions taken.
    The states should be a list of states the agent visit. The first should be the initial state and the last
    should be the won state. """
    # TODO: 4 lines missing.
    env = PacmanEnvironment(layout_str=map)#, render_mode='human')
    initial_x, _ = env.reset()
    agent = DynamicalProgrammingAgent(env, model=PacManNoGhost(N, initial_x))
    
    # conversion to optimal state and action list
    x_temp = initial_x
    actions, states = [], []
    for k in range(N):
        u = agent.pi(x_temp,k)
        states.append(x_temp)
        if x_temp.isWin(): break # if we win, the loop needs to break
        # we want the last state but not the last action
        actions.append(u)
        x_temp = x_temp.f(u)
    return actions, states


def no_ghosts():
    # Check the pacman_demo.py file for help on the GameState class and how to get started.
    # This function contains examples of calling your functions. However, you should use unitgrade to verify correctness.

    ## Problem 1: Lets try to go East. Run this code to see if the states you return looks sensible.
    states = go_east(east)
    for s in states:
        print(str(s))

    ## Problem 3: try the p_next function for a few empty environments. Does the result look sensible?
    x, _ = PacmanEnvironment(layout_str=east).reset()
    action = x.A()[0]
    print(f"Transitions when taking action {action} in map: 'east'")
    print(x)
    print(p_next(x, action))  # use str(state) to get a nicer representation.

    print(f"Transitions when taking action {action} in map: 'east2'")
    x, _ = PacmanEnvironment(layout_str=east2).reset()
    print(x)
    print(p_next(x, action))

    ## Problem 4
    print(f"Checking states space S_1 for k=1 in SS0tiny:")
    x, _ = PacmanEnvironment(layout_str=SS0tiny).reset()
    states = get_future_states(x, N=10)
    for s in states[1]: # Print all elements in S_1.
        print(s)
    print("States at time k=10, |S_10| =", len(states[10]))

    ## Problem 6
    N = 20  # Planning horizon
    action, states = shortest_path(east, N)
    print("east: Optimal action sequence:", action)

    action, states = shortest_path(datadiscs, N)
    print("datadiscs: Optimal action sequence:", action)

    action, states = shortest_path(SS0tiny, N)
    print("SS0tiny: Optimal action sequence:", action)


def one_ghost():
    # Win probability when planning using a single ghost. Notice this tends to increase with planning depth
    wp = []
    for n in range(10):
        wp.append(win_probability(SS1tiny, N=n))
    print(wp)
    print("One ghost:", win_probability(SS1tiny, N=12))


def two_ghosts():
    # Win probability when planning using two ghosts
    print("Two ghosts:", win_probability(SS2tiny, N=12))

if __name__ == "__main__":
    #no_ghosts()
    #one_ghost()
    two_ghosts()

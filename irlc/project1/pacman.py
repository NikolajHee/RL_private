# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from collections import defaultdict
from irlc import train
from irlc.ex02.dp_model import DPModel
from irlc.ex02.dp import DP_stochastic
from irlc.ex02.dp_agent import DynamicalProgrammingAgent
from irlc.pacman.pacman_environment import PacmanEnvironment
import numpy as np

# !s=east

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
#raise NotImplementedError("Put your own code here")
class PacManNoGhost(DPModel):
    def __init__(self, N, x0):
        super().__init__(N)
        self.S_save = get_future_states(x0, N)
    def f(self, x, u, w, k):
        return list(p_next(x,u))[0]
    def g(self, x, u, w, k):
        return 0 if (x.isWin()) else k
    def gN(self, x):
        return -1 if x.isWin() else 0
    def S(self, k):
        #print(self.S_save[k])
        return self.S_save[k]
    def A(self, x, k):
        return x.A()
    #def Pw(self, x, u, k):
class PacManOneGhost(DPModel):
    def __init__(self, N, x0):
        super().__init__(N)
        self.S_save = get_future_states(x0, N)

    def f(self, x, u, w, k):
        dict_ = p_next(x,u)
        #prob = list(dict_.values())
        states = list(dict_.keys())
        return states[w]

    def g(self, x, u, w, k):
        return 0

    def gN(self, x):
        #if x.isWin(): return -1
        #if x.isLose(): return 0
        return -1 if x.isWin() else 0

    def S(self, k):
        #print(self.S_save[k])
        return self.S_save[k]

    def A(self, x, k):
        return x.A()

    def Pw(self, x, u, k):
        dict_ = p_next(x,u)
        n = len(dict_.values())
        return {i: 1/n for (i,_) in zip(range(n), range(n))}
        # calculating p_next again?


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
    x_new = x.f(u) 
    if x_new.player() == 0: return {x_new: 1} # in case of no ghosts
    number_of_actions = len(x_new.A())
    uniform_distribution = [1/number_of_actions for _ in range(number_of_actions)]
    
    assert (sum(uniform_distribution) - 1) < 1e-8 # testing if they are approx equal to 100%
    
    return {x_new.f(i): j for (i,j) in zip(x_new.A(),uniform_distribution)}


def go_east(map): 
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
    #raise NotImplementedError("Return the list of states pacman will traverse if he goes east until he wins the map")
    env = PacmanEnvironment(layout_str=east)#, render_mode='human')
    x, info = env.reset()
    states = []
    x_new = x
    states.append(x_new)
    t = 0
    while ("East" in x_new.A()) and t < 10000:
        x_new = x_new.f('East')
        states.append(x_new)
        t += 1
    return states

def get_future_states(x, N): 
    # TODO: 4 lines missing.
    state_spaces = []
    state_spaces.append([x]) # S0

    # lacks the check for dubplicates
    #for i in range(N):
    #    state_spaces.append([[list(p_next(k,j))[0] for j in k.A()] for k in state_spaces[i]][0])
    
    #  # this works but not so cool
    for i in range(N):
        states = []
        for k in state_spaces[i]:
            # available actions:
                for u in k.A():
                    u_new_list = list(p_next(k,u).keys())
                    for u_new in u_new_list:
                        if u_new not in states: states.append(u_new)
        state_spaces.append(states)
    return state_spaces

def win_probability(map, N=10): 
    """ Assuming you get a reward of -1 on wining (and otherwise zero), the win probability is -J_pi(x_0). """
    # TODO: 5 lines missing.
    env = PacmanEnvironment(layout_str=map)
    initial_x, _ = env.reset()
    model = PacManOneGhost(N, initial_x)
    agent = DynamicalProgrammingAgent(env, model=model)

    win_probability = -agent.J[0][initial_x]
    #raise NotImplementedError("Return the chance of winning the given map within N steps or less.")
    return win_probability

def shortest_path(map, N=10): 
    """ If each move has a cost of 1, the shortest path is the path with the lowest cost.
    The actions should be the list of actions taken.
    The states should be a list of states the agent visit. The first should be the initial state and the last
    should be the won state. """
    # TODO: 4 lines missing.
    #model = DPModel(10)
    #states = go_east(map) # return list of states
    env = PacmanEnvironment(layout_str=map)#, render_mode='human')
    initial_x, _ = env.reset()
    model = PacManNoGhost(N, initial_x)
    agent = DynamicalProgrammingAgent(env, model=model)
    
    # conversion to optimal state and action list
    x_temp = initial_x
    actions, states = [], []
    for k in range(model.N):
        u = agent.pi(x_temp,k)
        states.append(x_temp)
        if x_temp.isWin(): break # if we win, the loop needs to break
        # we want the last state but not the last action
        actions.append(u)
        x_temp = x_temp.f(u)
        
    #print("J", agent.J)
    #raise NotImplementedError("Return the cost of the shortest path, the list of actions taken, and the list of states.")
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
    one_ghost()
    #two_ghosts()

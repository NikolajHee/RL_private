# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from collections import defaultdict
from irlc import train
from irlc.ex01.dp_model import DPModel
from irlc.ex02.dp import DP_stochastic
from irlc.ex02.dp_agent import DynamicalProgrammingAgent
from irlc.pacman.pacman_environment import GymPacmanEnvironment

#!s=east
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
% P  %
%G G.%
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
% P  %
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

# TODO: 29 lines missing.
raise NotImplementedError("Put your own code here")

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
    raise NotImplementedError("Return a dictionary {.., xp: p, ..} where xp is a possible next state and p the probability")
    return states


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
    # TODO: 4 lines missing.
    raise NotImplementedError("Return the list of states pacman will traverse if he goes east until he wins the map")
    return states

def get_future_states(x, N): 
    # TODO: 4 lines missing.
    raise NotImplementedError("return a list-of-list of future states [S_0,\dots,S_N]. Each S_k is a state space, i.e. a list of GameState objects.")
    return state_spaces

def win_probability(map, N=10): 
    # TODO: 3 lines missing.
    raise NotImplementedError("Return the chance of winning the given map within N steps or less.")
    return -J[0][env.reset()]

def shortest_path(map, N=10): 
    # TODO: 7 lines missing.
    raise NotImplementedError("Return the cost of the shortest path, the list of actions taken, and the list of states.")
    return J[0][env.reset()], list(traj.action), list(traj.state)


def no_ghosts():
    # Check the pacman_demo.py file for help on the GameState class and how to get started.
    ## Question 1: Lets try to go East. Run this code to see if the states you return looks sensible.
    states = go_east(east)
    for s in states:
        print(str(s))

    x = GymPacmanEnvironment(layout_str=east).reset()
    action = x.A()[0]
    print(f"Transitions when taking action {action} in map:")
    print(x)
    print(p_next(x, action))

    print(f"Transitions when taking action {action} in map:")
    x = GymPacmanEnvironment(layout_str=east2).reset()
    print(x)
    print(p_next(x, action))

    ## Question 3
    print(f"Checking states space S_1 for k=1 in SS0tiny:")
    x = GymPacmanEnvironment(layout_str=SS0tiny).reset()
    states = get_future_states(x, N=10)
    for s in states[1]:
        print(s)
    print("States at time k=10, |S_10| =", len(states[10]))

    ## Question 4
    N = 20 # Planning horizon
    minimum_cost, action, states = shortest_path(east, N)
    print("Optimal cost is", minimum_cost, "optimal action sequence:", action)

    minimum_cost, action, states = shortest_path(datadiscs, N)
    print("Optimal cost is", minimum_cost, "optimal action sequence:", action)

    minimum_cost, action, states = shortest_path(SS0tiny, N)
    print("Optimal cost is", minimum_cost, "optimal action sequence:", action)


def one_ghost():
    # Win probability when planning using a single ghost. Notice this increases over time:
    wp = []
    for n in range(5):
        wp.append(win_probability(SS1tiny, N=n))

    print(wp)
    print(win_probability(SS1tiny, N=12))

def two_ghost():
    # Win probability when planning using two ghosts
    print(win_probability(SS2tiny, N=12))

if __name__ == "__main__":
    no_ghosts()
    one_ghost()
    two_ghost()

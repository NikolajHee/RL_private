# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf).
"""
from irlc import savepdf
from irlc.ex09.mdp import MDP
from irlc.ex09.value_iteration import value_iteration
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# These are the game rules of the sarlac: If you land on a state s in the dictionary, you are teleported to rules[s].
rules = {
        2: 16,
        4: 8,
        7: 21,
        10: 3,
        12: 25,
        14: 1,
        17: 27,
        19: 5,
        22: 3,
        23: 32,
        24: 44,
        26: 44,
        28: 38,
        30: 18,
        33: 48,
        35: 11,
        36: 34,
        40: 53,
        41: 29,
        42: 9,
        45: 51,
        47: 31,
        50: 25,
        52: 38,
        55: -1,
    }

def game_rules(rules : dict, state : int, roll : int) -> int: 
    """ Compute the next state given the game rules in 'rules', the current state 'state', and the roll
    which can be roll = 1, 2, 3, 4, 5, 6.
    The output should be -1 in case the game terminates, and otherwise the function should return the next state
    as an integer. Read the description of the project for examples on the rules. """
    # TODO: 4 lines missing.
    new_state = state + roll
    if new_state == 55: return -1
    if new_state > 55: new_state = 55 - (new_state - 55)
    if new_state in rules: new_state = rules[new_state]
    return new_state

# TODO: 19 lines missing.
class Sarlacc(MDP):
    def __init__(self, rules):
        super().__init__(initial_state=0)
        self.rules = rules
    
    def A(self, state):
        return set(range(1,7))
    
    def Psr(self, state, action):
        PSR = defaultdict(lambda: 0)
        for (s) in [game_rules(self.rules, state, a) for a in self.A(state)]: PSR[(s,1)] += (1/6)
        return PSR

    def is_terminal(self, state):
        return state == -1
    
    
def sarlacc_return(rules : dict, gamma : float) -> dict: 
    """ Compute the value-function using a discount of gamma and the game rules 'rules'.
    Result should be reasonable accurate.

    The value you return should be a dictionary v, so that v[state] is the value function in that state.
    (i.e., the standard output format of the value_iteration function).

    Hints:
        * One way to solve this problem is to create a MDP-class (see for instance the Gambler-problem in week 9)
        and use the value_iteration function from week 9 to solve the problem. But I don't think the problem
        is much harder to solve by just writing your own value-iteration method as in (SB18).
    """
    # TODO: 2 lines missing.
    _, v = value_iteration(mdp = Sarlacc(rules), gamma=gamma)
    return v


if __name__ == "__main__":
    """ 
    Rules for the snakes and ladder game: 
    The player starts in square s=0, and the game terminates when the player is in square s = 55. 
    When a player reaches the base of a ladder he/she climbs it, and when they reach a snakes mouth of a snake they are translated to the base.
    When a player overshoots the goal state they go backwards from the goal state by the amount of moves they overshoot with.
    
    A few examples (using the rules in the 'rules' dictionary in this file):
    If the player is in position s=0 (start)
    > roll 2: Go to state s=16 (using the ladder)
    > roll 3: Go to state s=3. 

    Or if the player is in state s=54
    > Roll 1: Win the game
    > Roll 2: stay in 54
    > Roll 3: Go to 53
    > Roll 4: Go to 38    
    """
    # Test the game rules:
    #for roll in [1, 2, 3, 4, 5, 6]:
    #    print(f"In state s=0 (start), using roll {roll}, I ended up in ", game_rules(rules, 0, roll))
    # Test the game rules again:
    #for roll in [1, 2, 3, 4, 5, 6]:
    #    print(f"In state s=54, using roll {roll}, I ended up in ", game_rules(rules, 54, roll))

    # Compute value function with the ordinary rules.
    V_rules = sarlacc_return(rules, gamma=1)
    # Compute value function with no rules, i.e. with an empty dictionary except for the winning state:
    V_norule = sarlacc_return({55: -1}, gamma=1)
    print("Time to victory when there are no snakes/ladders", V_norule[0])
    print("Time to victory when there are snakes/ladders", V_rules[0])

    # Make a plot of the value-functions (optional).
    width = .4
    def v2bar(V):
        k, x = zip(*V.items())
        return np.asarray(k), np.asarray(x)

    plt.figure(figsize=(10,5))
    plt.grid()
    k,x = v2bar(V_norule)
    plt.bar(k-width/2, x, width=width, label="No rules")

    k, x = v2bar(V_rules)
    plt.bar(k + width / 2, x, width=width, label="Rules")
    plt.legend()
    plt.xlabel("Current tile")
    plt.ylabel("Moves remaining")
    savepdf('sarlacc_value_function')
    plt.show()

# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc import savepdf, train
from irlc.ex09.mdp import MDP
from irlc.ex09.value_iteration import value_iteration
import matplotlib.pyplot as plt
import numpy as np
from irlc.project3i.sarlacc import sarlacc_return, rules, game_rules, SnakesMDP
from irlc.ex09.value_iteration_agent import ValueIterationAgent
from irlc.ex09.mdp import MDP2GymEnv

if __name__ == "__main__":
    """ 
    Rules for the snakes and ladder game: 
    The player starts in square s=0, and the game terminates when the player is in square s = 55. 
    When a player reaches the base of a ladder he/she climbs it, and when they reach a snakes mouth of a snake they are translated to the base.
    When a player overshoots the goal state they go backwards from the goal state. 

    A few examples:    
    If the player is in position s=0 (start)
    > roll 3: Go to state s=3. 
    > roll 2: Go to state s=16 (using the ladder)

    Or if the player is in state s=54
    > Roll 1: Win the game
    > Roll 2: stay in 54
    > Roll 3: Go to 53
    > Roll 4: Go to 52    
    """

    V_norule = sarlacc_return({55: -1}, gamma=1)
    V_rules = sarlacc_return(rules, gamma=1)
    print("Time to victory when there are no snakes/ladders", V_norule[0])
    print("Time to victory when there are snakes/ladders", V_rules[0])

    # TODO: 17 lines missing.
    raise NotImplementedError("Create plot of value functions (I suggest a bar-plot).")

    env = MDP2GymEnv(SnakesMDP(rules))
    I wrote the plot-code here.
    # TODO: 3 lines missing.
    raise NotImplementedError("use the train()-function to simulate a bunch of games.")

    avg = np.mean(z)                # this and --
    pct95 = np.percentile(z, q=95)  # This might be of help (define z above).
    print(f"Average game length was: {avg:.1f} and 5% of the games last longer than {pct95:.1f} moves :-(")  

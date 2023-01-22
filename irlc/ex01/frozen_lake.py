# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
Uses https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
Check out the URL for details
"""
import gym
import matplotlib.pyplot as plt
import numpy as np
from irlc import savepdf
from irlc import Agent, train
from gym.envs.toy_text.frozen_lake import RIGHT, DOWN  # The down and right-actions; may be relevant.

class FrozenAgentDownRight(Agent):
    """
    A silly agent which, in stage k, moves down if k is even and otherwise moves right. You only have to implement the policy-function.
    """
    def pi(self, s, k, info=None): 
        # TODO: 3 lines missing.
        raise NotImplementedError("Implement function body")
        return RIGHT

def to_s(row, col, ncol):
    """ Converts from (row,col) to linear indexing. This function is taken from the frozen_lake environment """
    return row * ncol + col

def to_rc(s, ncol): 
    """
    inverse of the to_s function from

    https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py

    Given s, should return row, col such that when we call to_s(row, col, ncol) we get the same s back.
    """
    # TODO: 2 lines missing.
    raise NotImplementedError("Implement function body")
    return row, col

if __name__ == "__main__":
    np.random.seed(42) # Set the seed to get deterministic results.
    """
    Openai gym uses something called wrappers to do various useful pre-processing steps. This means that the actual env
    can be accessed as the variable env.env, whereas env represents a time-limited "wrapper" class.
    
    Asides this, remember that actions are now called a, states are called x, and we use reward rather than cost.
    
    At any rate, let's make our frozen lake environments.
    """
    # render_mode = 'human' gives us a visualization.
    env = gym.make("FrozenLake-v1", render_mode='human')
    """This function can be used to render the current environment"""
    env.reset()  # reset to starting position and open the visualization.
    import time
    time.sleep(1) # Show the pretty screenshot for 1 second.
    env.close()
    env = gym.make("FrozenLake-v1") # This is much faster.
    """
    Get average reward
    """
    T = 2000
    stats, _ = train(env, Agent(env), num_episodes=T, verbose=False)
    Ravg = np.mean([stat['Accumulated Reward'] for stat in stats])
    print("Average reward of random policy", Ravg) 

    stats, _ = train(env, FrozenAgentDownRight(env), num_episodes=T,verbose=False)
    Ravg = np.mean([stat['Accumulated Reward'] for stat in stats])
    print("Average reward of down_right policy", Ravg) 

    """ Check code to convert states from linear to (i,j) indices """
    s = 5
    ro,co = to_rc(s, ncol=env.ncol)
    s2 = to_s(ro, co, ncol=env.ncol)
    print("These should be equal: s", s, "s2", s2)

    """ plot trajectories """
    stats, trajectories = train(env, Agent(env), num_episodes=50, return_trajectory=True)
    for trajectory in trajectories:
        p = [to_rc(s, ncol=env.env.ncol) for s in trajectory.state]
        I, J = zip(*p)
        wgl = 0.1
        plt.plot(np.random.randn( len(I) )*wgl + I, np.random.randn( len(J) )*wgl + J, 'k.-')
    plt.xlabel('Row')
    plt.ylabel('Col')
    plt.gca().invert_yaxis()
    savepdf("frozen_random.pdf")
    plt.show()

    """ Last part: Generate and plot 50 trajectories from the FrozenDownRightAgent  """
    # TODO: 6 lines missing.
    raise NotImplementedError("Plot 50 trajectories from the FrozenDownRightAgent")
    plt.xlabel('Row')
    plt.ylabel('Col')
    plt.gca().invert_yaxis()
    savepdf("frozen_downright.pdf")
    plt.show()

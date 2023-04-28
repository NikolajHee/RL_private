# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from irlc.ex11.q_agent import QAgent
from irlc.gridworld.gridworld_environments import GridworldEnvironment, grid_bridge_grid
from irlc import train
from irlc.ex09.rl_agent import TabularQ

# A simple UCB action-selection problem (basic problem)
very_basic_grid = [['#',1, '#'],
                    [1, 'S', 2],
                    ['#',1, '#']]


# TODO: 21 lines missing.
class UCBAgent(QAgent):
    def __init__(self, env, gamma=1.0, c = 0.1, alpha=0.5, epsilon=0.1):
        self.alpha = alpha
        super().__init__(env, gamma, epsilon)
        self.N = TabularQ(env)
        self.c = c

    def pi(self, s, k, info=None):         
        actions, QA = self.Q.get_Qs(s, info_s = info)
        _, Ns = self.N.get_Qs(s, info_s = info)
        t = np.sum(Ns)
        QA = np.array(QA)
        Ns = np.array(Ns)
        a_ = []
        for a in actions:
            if self.N[s,a] == 0: return a
            a_.append(QA[a] + self.c * np.sqrt(np.log(t)/(Ns[a])))
        a_star = actions[np.argmax(a_)]
        return a_star

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None): 
        if not done:
            a_star = self.Q.get_optimal_action(sp, info_sp)
            self.N[s,a] += 1
        self.Q[s,a] +=  self.alpha * (r + self.gamma *(0 if done else self.Q[sp,a_star]) - self.Q[s,a])


def get_ucb_actions(layout : list, alpha : float, c : float, episodes : int, plot=False) -> list: 
    """ Return the sequence of actions the agent tries in the environment with the given layout-string when trained over 'episodes' episodes.
    To create an environment, you can use the line:

    > env = GridworldEnvironment(layout)

    See also the demo-file.

    The 'plot'-parameter is optional; you can use it to add visualization using a line such as:

    if plot:
        env = GridworldEnvironment(layout, render_mode='human')

    Or you can just ignore it. Make sure to return the truncated action list (see the rebels_demo.py-file or project description).
    In other words, the return value should be a long list of integers corresponding to actions:
    actions = [0, 1, 2, ..., 1, 3, 2, 1, 0, ...]
    """
    # TODO: 6 lines missing.
    env = GridworldEnvironment(layout)
    env.reset()
    if plot:
        env = GridworldEnvironment(layout, render_mode='human')    
    agent = UCBAgent(env, alpha=alpha, c=c)
    _, trajectories = train(env, agent, num_episodes=episodes, return_trajectory=True)
    actions = [t for traj in trajectories for t in traj.action[:-1]]
    return actions

if __name__ == "__main__":
    #actions = get_ucb_actions(very_basic_grid, alpha=0.1, c=5, episodes=4, plot=False)
    #print("Number of actions taken", len(actions))
    #print("List of actions taken over 4 episodes", actions)

    #from irlc.gridworld.gridworld_environments import grid_bridge_grid
    actions = get_ucb_actions(grid_bridge_grid, alpha=0.1, episodes=2, c=2, plot=False)
    print("Number of actions taken", len(actions))
    print("List of actions taken over 2 episodes", actions)

    # actions = get_ucb_actions(very_basic_grid, alpha=0.1, c=5, episodes=8, plot=False)
    # print("Number of actions taken", len(actions))
    # print("Actions taken over 8 episodes", actions)

    # actions = get_ucb_actions(very_basic_grid, alpha=0.1, c=5, episodes=9, plot=False)
    # print("Number of actions taken", len(actions))
    # print("Actions taken over 9 episodes", actions) # In this particular case, you can also predict the 9th action. Why?

    # # Simulate 100 episodes. This should solve the problem.
    # actions = get_ucb_actions(very_basic_grid, alpha=0.1, c=5, episodes=100, plot=False)
    # print("Basic: Actions taken over 100 episodes", actions)

    # # Simulate 100 episodes for the bridge-environment. The UCB-based method should solve the environment without being overly sensitive to c.
    # # You can compare your result with the Q-learning agent in the demo, which performs horribly.
    # actions = get_ucb_actions(grid_bridge_grid, alpha=0.1, c=5, episodes=300, plot=False)
    # print("Bridge: Actions taken over 300 episodes. The agent should solve the environment:", actions)

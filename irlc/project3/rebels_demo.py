# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from irlc import train, VideoMonitor, Agent, PlayWrapper
from irlc.gridworld.gridworld_environments import GridworldEnvironment, grid_bridge_grid
from irlc.project3.rebels import very_basic_grid
from irlc.ex11.q_agent import QAgent

if __name__ == "__main__":
    np.random.seed(42) # Fix the seed for reproduciability
    env = GridworldEnvironment(very_basic_grid) # Create an environment
    env = VideoMonitor(env) # Create a visualization
    env.reset()                   # Reset (to set up the visualization)
    env.savepdf("rebels_basic")   # Save a snapshot of the starting state
    agent = Agent(env) # A random agent.
    # agent = PlayWrapper(agent, env) # Uncomment these three lines to play in 'env' environment:
    stats, trajectories = train(env, agent, num_episodes=16, return_trajectory=True)
    env.close()
    print("Trajectory 0: States traversed", trajectories[0].state, "actions taken", trajectories[0].action) 
    print("Trajectory 1: States traversed", trajectories[1].state, "actions taken", trajectories[1].action)
    all_actions = [t.action[:-1] for t in trajectories] # Concatenate all action sequence excluding the last dummy-action.
    print("All actions taken in 16 episodes, excluding the terminal (dummy) action", all_actions) 
    # Note the last list is of length 20 -- this is because the environment will always terminate after two actions,
    # and since we discard the last (dummy) action we get 20 actions.
    # In general, the list of actions will be longer, as only the last action should be discarded (as in the code above).

    # A more minimalistic example to plot the bridge-grid environment
    bridge_env = VideoMonitor(GridworldEnvironment(grid_bridge_grid))
    bridge_env.reset()
    bridge_env.savepdf("rebels_bridge")
    bridge_env.close()

    # The following code will simulate a Q-learning agent for 3000 (!) episodes and plot the Q-functions.
    np.random.seed(42)  # Fix the seed for reproduciability
    env = GridworldEnvironment(grid_bridge_grid)
    agent = QAgent(env, alpha=0.1, epsilon=0.2, gamma=1)
    """ Uncomment the next line to play in the environment. 
    Use the space-bar to let the agent take an action, p to unpause, and otherwise use the keyboard arrows """
    # agent = PlayWrapper(agent, env)
    train(env, agent, num_episodes=3000) # Train for 3000 episodes. Surely the rebels must be found by now!
    bridge_env = VideoMonitor(env, agent=agent)
    bridge_env.reset()
    bridge_env.savepdf("rebels_bridge_Q")
    bridge_env.close()

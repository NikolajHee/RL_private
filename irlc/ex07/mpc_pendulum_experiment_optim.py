# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex01.agent import train
from irlc.ex07.learning_agent_mpc_optimize import MPCLearningAgentLocalOptimize
from irlc import plot_trajectory, main_plot
import matplotlib.pyplot as plt
from irlc.ex07.mpc_pendulum_experiment_lqr import mk_mpc_pendulum_env

L = 12

def main_pendulum():
    env_pendulum = mk_mpc_pendulum_env(Tmax=10, render_mode='human')
    """ Run Local Optimization/MPC agent using the parameters
    L = 12 
    neighboorhood_size=50
    min_buffer_size=50 
    """
    agent = MPCLearningAgentLocalOptimize(env_pendulum, horizon_length=L, neighbourhood_size=50, min_buffer_size=50)

    experiment_name = f"pendulum{L}"
    stats, trajectories = train(env_pendulum, agent, experiment_name=experiment_name, num_episodes=16,return_trajectory=True)
    plt.show()
    for k in range(len(trajectories)):
        plot_trajectory(trajectories[k], env_pendulum)
        plt.title(f"Trajectory {k}")
        plt.show()

    env_pendulum.close()
    main_plot(experiment_name)
    plt.show()

if __name__ == "__main__":
    main_pendulum()

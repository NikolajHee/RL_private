# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex04.model_pendulum import GymSinCosPendulumEnvironment
from irlc.ex04.cost_discrete import DiscreteQRCost
from irlc.ex01.agent import train
from irlc.ex07.lqr_learning_agents import MPCLocalLearningLQRAgent
from irlc import plot_trajectory, main_plot
import matplotlib.pyplot as plt
import numpy as np


def mk_mpc_pendulum_env(Tmax=10, render_mode=None):
    """
    Initialize pendulum model suitable for MPC learning.

    If you try to replicate this experiment for another environment, please note I had to tweak the
    parameters quite a lot to make things work.
    """
    env_pendulum = GymSinCosPendulumEnvironment(Tmax=Tmax, dt=0.08, transform_actions=False, render_mode=render_mode)
    model = env_pendulum.discrete_model
    Q = np.eye(model.state_size)
    Q = Q * 0
    Q[1, 1] = 1.0  # up-coordinate
    q = np.zeros((model.state_size,))
    q[1] = -1

    cost2 = DiscreteQRCost(Q=np.eye(model.state_size), R=np.eye(model.action_size)) * 0.03
    cost2 = cost2 + cost2.goal_seeking_cost(Q=Q, x_target=model.x_upright) * 1
    # cost2 = goal_seeking_qr_cost(model, Q=Q, x_target=model.x_upright) * 1
    model.cost = cost2
    return env_pendulum

L = 12
def main_pendulum_lqr(Tmax=10):

    """ Run Local LQR/MPC agent using the parameters
    L = 12  
    neighboorhood_size = 50
    min_buffer_size = 50 
    """
    env_pendulum = mk_mpc_pendulum_env(render_mode='human')
    # agent = .... (instantiate agent here)
    agent = MPCLocalLearningLQRAgent(env_pendulum, horizon_length=L, neighbourhood_size=50, min_buffer_size=50)


    experiment_name = f"pendulum{L}_lqr"
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
    main_pendulum_lqr()

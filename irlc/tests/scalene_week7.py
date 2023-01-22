# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from irlc import train


# from scalene import scalene_profiler
# scalene_profiler.start()


def test_pendulum_swingup():
    """ Pendulum swingup task """
    tl = 1 - np.cos(np.pi / 180 * 20)

    from irlc.ex07.mpc_pendulum_experiment_optim import mk_mpc_pendulum_env
    env_pendulum = mk_mpc_pendulum_env(Tmax=10, render_mode="human")
    L = 12
    up = 100
    from irlc.ex07.learning_agent_mpc_optimize import MPCLearningAgentLocalOptimize
    agent = MPCLearningAgentLocalOptimize(env_pendulum, horizon_length=L, neighbourhood_size=50, min_buffer_size=50)
    # from irlc import VideoMonitor
    # env_pendulum = VideoMonitor(env_pendulum)
    for _ in range(10):
        stats, trajectories = train(env_pendulum, agent, num_episodes=1, return_trajectory=True)
        cos = trajectories[0].state[:, 1]
        up = np.abs((cos - 1)[int(len(cos) * .8):])

        if np.all(max(up) < tl):
            print("Win!!!")
            break

    # Turn profiling off
    # self.assertTrue(all(up < tl))

test_pendulum_swingup()
# scalene_profiler.stop()

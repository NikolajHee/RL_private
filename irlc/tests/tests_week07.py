# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, Report
import irlc
from irlc.ex04.model_boing import BoingEnvironment
from irlc import train
import numpy as np

""" Week 7 Questions """

class LearningLQRAgentQuestion(UTestCase):
    """ LearningLQRAgent on Boing problem """
    # class TrajectoryItem(LearningItem):
    def test_lqr_agent(self):
        from irlc.ex07.lqr_learning_agents import LearningLQRAgent
        env = BoingEnvironment(output=[10, 0])
        agent = LearningLQRAgent(env)
        from irlc.ex07.lqr_learning_agents import boing_experiment
        stats, trajectories = boing_experiment(env, agent, num_episodes=4, plot=False)
        self.assertL2(stats[-1]['Accumulated Reward'], tol=0.4)
        self.assertL2(trajectories[-1].state[-1], tol=0.05)


class MPCLearningAgentQuestion(UTestCase):
    """ MPCLearningAgent on Boing problem """
    # class TrajectoryItem(LearningItem):
    def test_lqr_agent(self):
        from irlc.ex07.lqr_learning_agents import MPCLearningAgent
        env = BoingEnvironment(output=[10, 0])
        agent = MPCLearningAgent(env)
        from irlc.ex07.lqr_learning_agents import boing_experiment
        stats, trajectories = boing_experiment(env, agent, num_episodes=4, plot=False)
        self.assertL2(stats[-1]['Accumulated Reward'], tol=0.8)
        self.assertL2(trajectories[-1].state[-1], tol=0.05)

class MPCLocalLearningLQRAgent(UTestCase):
    """ MPCLocalLearningLQRAgent on Boing problem """

    def test_local_lqr_agent(self):
        from irlc.ex07.lqr_learning_agents import MPCLocalLearningLQRAgent
        env = BoingEnvironment(output=[10, 0])
        agent = MPCLocalLearningLQRAgent(env)
        from irlc.ex07.lqr_learning_agents import boing_experiment
        stats, trajectories = boing_experiment(env, agent, num_episodes=3, plot=False)
        self.assertL2(stats[-1]['Accumulated Reward'], tol=2200)
        self.assertL2(trajectories[-1].state[-1], tol=0.75)

class MPCLearningAgentLocalOptimizeQuestion(UTestCase):
    """ MPCLearningAgentLocalOptimize on Boing problem """
    def test_local_lqr_agent(self):
        from irlc.ex07.learning_agent_mpc_optimize import MPCLearningAgentLocalOptimize
        env = BoingEnvironment(output=[10, 0])
        agent = MPCLearningAgentLocalOptimize(env)
        from irlc.ex07.lqr_learning_agents import boing_experiment
        stats, trajectories = boing_experiment(env, agent, num_episodes=2, plot=False)
        self.assertL2(stats[-1]['Accumulated Reward'], tol=1600)
        self.assertL2(trajectories[-1].state[-1], tol=0.75) # Bump to 0.3.

class MPCLocalAgentExactDynamicsQuestion(UTestCase):
    """ MPCLocalAgentExactDynamics on Boing problem """

    # class TrajectoryItem(LearningItem):
    def test_local_lqr_agent(self):
        # from irlc.ex07.lqr_learning_agents import MPCLocalAgentExactDynamics
        from irlc.ex07.learning_agent_mpc_optimize import MPCLocalAgentExactDynamics

        env = BoingEnvironment(output=[10, 0])
        agent = MPCLocalAgentExactDynamics(env)
        from irlc.ex07.lqr_learning_agents import boing_experiment
        stats, trajectories = boing_experiment(env, agent, num_episodes=3, plot=False)
        self.assertL2(stats[-1]['Accumulated Reward'], tol=0.05)
        self.assertL2(trajectories[-1].state[-1], tol=0.05)

class MPCLearningPendulum(UTestCase):
    """ MPCLearningAgentLocalOptimize on pendulum problem """
    def test_pendulum_swingup(self):
        """ Pendulum swingup task """
        tl = 1 - np.cos(np.pi / 180 * 20)
        # from scalene import scalene_profiler
        #
        # # Turn profiling on
        # scalene_profiler.start()


        from irlc.ex07.mpc_pendulum_experiment_optim import mk_mpc_pendulum_env
        env_pendulum = mk_mpc_pendulum_env(Tmax=10, render_mode=None)
        L = 12
        up = 100
        from irlc.ex07.learning_agent_mpc_optimize import MPCLearningAgentLocalOptimize
        agent = MPCLearningAgentLocalOptimize(env_pendulum, horizon_length=L, neighbourhood_size=50, min_buffer_size=50)
        # from irlc import VideoMonitor
        # env_pendulum = VideoMonitor(env_pendulum)
        for _ in range(15):
            stats, trajectories = train(env_pendulum, agent, num_episodes=1, return_trajectory=True)
            cos = trajectories[0].state[:, 1]
            up = np.abs((cos - 1)[int(len(cos) * .8):])

            if np.all(max(up) < tl):
                break

        # Turn profiling off
        # scalene_profiler.stop()
        self.assertTrue(all(up < tl))

class Week07Tests(Report):
    title = "Tests for week 07"
    pack_imports = [irlc]
    individual_imports = []
    questions = [
        (LearningLQRAgentQuestion, 10),             # ok # About 2.75 minutes.
        (MPCLearningAgentQuestion, 5),              # ok
        # (MPCLearningAgentLocalOptimizeQuestion, 10),# ok This one takes bout 3.5 minutse.
        (MPCLocalAgentExactDynamicsQuestion, 5),    # ok
        (MPCLocalLearningLQRAgent, 10),  # ok
        (MPCLearningPendulum, 10),                  # ok About 2.75 minutes.
                 ]



if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week07Tests())

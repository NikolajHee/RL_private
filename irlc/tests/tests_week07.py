# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import Report
import irlc
from irlc.ex01.frozen_lake import FrozenAgentDownRight
import gym
from unitgrade import UTestCase
from irlc.ex01.inventory_environment import InventoryEnvironment, simplified_train, RandomAgent
from unitgrade import Capturing2
from irlc import train, Agent
import numpy as np
from gym.envs.toy_text.frozen_lake import RIGHT, DOWN  # The down and right-actions; may be relevant.
from irlc.ex01.pacman_hardcoded import GoAroundAgent, layout
from irlc.pacman.pacman_environment import GymPacmanEnvironment
from irlc import Agent, VideoMonitor, train, PlayWrapper


from unitgrade import UTestCase, Report #, QPrintItem
# from unitgrade_v1.unitgrade import Capturing
import irlc
from irlc.car.car_model import CarEnvironment
from irlc.ex04.pid_car import PIDCarAgent
from irlc.ex04.model_boing import BoingEnvironment
from irlc import train
import numpy as np

""" Week 7 Questions """
# class Week7Group(QuestionGroup):
#     title = "MPC and learning-MPC"
#
# class BoingQuestion_(QuestionGroup):
#     def make_agent(self):
#         return None
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#
# class BoingItem_(QPrintItem):
#     def compute_answer_print(self):
#         pass

# class LearningItem(QPrintItem):
#     tol = 0.05
#     def get_agent(self, env):
#         return None
#
#     def compute_answer_print(self):
#         env = BoingEnvironment(output=[10, 0])
#         agent = self.get_agent(env)
#         from irlc.ex07.lqr_learning_agents import boing_experiment
#         stats, trajectories = boing_experiment(env, agent, num_episodes=4, plot=False)
#         return (stats[-1]['Accumulated Reward'], trajectories[-1].state[-1])
#
#     def process_output(self, res, txt, numbers):
#         return np.abs(np.concatenate( (np.asarray(res[0]).reshape( (1,) ), res[1] )))+1

class LearningLQRAgentQuestion(UTestCase):
    """ LearningLQRAgent on Boing problem """
    # class TrajectoryItem(LearningItem):
    def test_lqr_agent(self):
        from irlc.ex07.lqr_learning_agents import LearningLQRAgent
        env = BoingEnvironment(output=[10, 0])
        agent = LearningLQRAgent(env)
        from irlc.ex07.lqr_learning_agents import boing_experiment
        stats, trajectories = boing_experiment(env, agent, num_episodes=4, plot=False)
        self.assertL2(stats[-1]['Accumulated Reward'], tol=0.2)
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

    # class TrajectoryItem(LearningItem):
    def test_local_lqr_agent(self):
        from irlc.ex07.lqr_learning_agents import MPCLocalLearningLQRAgent
        env = BoingEnvironment(output=[10, 0])
        agent = MPCLocalLearningLQRAgent(env)
        from irlc.ex07.lqr_learning_agents import boing_experiment
        stats, trajectories = boing_experiment(env, agent, num_episodes=4, plot=False)
        self.assertL2(stats[-1]['Accumulated Reward'], tol=500)
        self.assertL2(trajectories[-1].state[-1], tol=0.5)

class MPCLearningAgentLocalOptimizeQuestion(UTestCase):
    """ MPCLearningAgentLocalOptimize on Boing problem """

    # class TrajectoryItem(LearningItem):
    def test_local_lqr_agent(self):
        from irlc.ex07.lqr_learning_agents import MPCLearningAgentLocalOptimize
        env = BoingEnvironment(output=[10, 0])
        agent = MPCLearningAgentLocalOptimize(env)
        from irlc.ex07.lqr_learning_agents import boing_experiment
        stats, trajectories = boing_experiment(env, agent, num_episodes=4, plot=False)
        self.assertL2(stats[-1]['Accumulated Reward'], tol=500)
        self.assertL2(trajectories[-1].state[-1], tol=0.12)




    # def get_agent(self, env):
    #     from irlc.ex07.lqr_learning_agents import MPCLearningAgent
    #     return MPCLearningAgent(env)

# class LearningLQRAgentQuestion(QuestionGroup):
#     title = "MPCLearningAgent on Boing problem"
#     class TrajectoryItem(LearningItem):
#         def get_agent(self, env):
#             from irlc.ex07.lqr_learning_agents import MPCLearningAgent
#             return MPCLearningAgent(env)

# class MPCLearningAgentQuestion(QuestionGroup):
#     title = "MPCLocalLearningLQRAgent on Boing problem"
#     tol = 0.1
#     class TrajectoryItem(LearningItem):
#         def get_agent(self, env):
#             from irlc.ex07.lqr_learning_agents import MPCLocalLearningLQRAgent
#             return MPCLocalLearningLQRAgent(env)
#         def process_output(self, res, txt, numbers):
#             return np.abs(res[1]) + 1

# class MPCLearningAgentLocalOptimizeQuestion(QuestionGroup):
#     title = "MPCLearningAgentLocalOptimize on Boing problem"
#     tol = 0.05
#     class TrajectoryItem(LearningItem):
#         def get_agent(self, env):
#             from irlc.ex07.lqr_learning_agents import MPCLearningAgentLocalOptimize
#             return MPCLearningAgentLocalOptimize(env)
#
# class MPCLocalAgentExactDynamicsQuestion(QuestionGroup):
#     title = "MPCLocalAgentExactDynamics on Boing problem"
#     class TrajectoryItem(LearningItem):
#         def get_agent(self, env):
#             from irlc.ex07.lqr_learning_agents import MPCLocalAgentExactDynamics
#             return MPCLocalAgentExactDynamics(env)

class MPCLocalAgentExactDynamicsQuestion(UTestCase):
    """ MPCLocalAgentExactDynamics on Boing problem """

    # class TrajectoryItem(LearningItem):
    def test_local_lqr_agent(self):
        from irlc.ex07.lqr_learning_agents import MPCLocalAgentExactDynamics
        env = BoingEnvironment(output=[10, 0])
        agent = MPCLocalAgentExactDynamics(env)
        from irlc.ex07.lqr_learning_agents import boing_experiment
        stats, trajectories = boing_experiment(env, agent, num_episodes=4, plot=False)
        self.assertL2(stats[-1]['Accumulated Reward'], tol=0.05)
        self.assertL2(trajectories[-1].state[-1], tol=0.05)

class MPCLearningPendulum(UTestCase):
    """ MPCLearningAgentLocalOptimize on pendulum problem """
    def test_pendulum_swingup(self):
        """ Pendulum swingup task """
        tl = 1 - np.cos(np.pi / 180 * 20)

        from irlc.ex07.mpc_pendulum_experiment import mk_mpc_pendulum_env
        env_pendulum = mk_mpc_pendulum_env(Tmax=10)
        L = 12
        up = 100
        from irlc.ex07.lqr_learning_agents import MPCLearningAgentLocalOptimize
        agent = MPCLearningAgentLocalOptimize(env_pendulum, horizon_length=L, neighbourhood_size=50, min_buffer_size=50)
        # from irlc import VideoMonitor
        # env_pendulum = VideoMonitor(env_pendulum)
        for _ in range(10):
            stats, trajectories = train(env_pendulum, agent, num_episodes=1, return_trajectory=True)
            cos = trajectories[0].state[:, 1]
            up = np.abs((cos - 1)[int(len(cos) * .8):])

            if np.all(max(up) < tl):
                break
        self.assertTrue(all(up < tl))


# class MPCLearningPendulum(QuestionGroup):
#     title = "MPCLearningAgentLocalOptimize on pendulum problem"
#     class PendulumItem(QPrintItem):
#         title = "Pendulum swingup task"
#         tl = 1-np.cos( np.pi/180 * 20)
#         def compute_answer_print(self):
#             from irlc.ex07.mpc_pendulum_experiment import mk_mpc_pendulum_env
#             env_pendulum = mk_mpc_pendulum_env(Tmax=10)
#             L = 12
#             up = 100
#             from irlc.ex07.lqr_learning_agents import MPCLearningAgentLocalOptimize
#             agent = MPCLearningAgentLocalOptimize(env_pendulum, horizon_length=L, neighbourhood_size=50, min_buffer_size=50)
#             # from irlc import VideoMonitor
#             # env_pendulum = VideoMonitor(env_pendulum)
#             for _ in range(7):
#                 stats, trajectories = train(env_pendulum, agent, num_episodes=1, return_trajectory=True)
#                 cos = trajectories[0].state[:,1]
#                 up = np.abs(  (cos - 1)[ int(len(cos) * .8): ] )
#
#                 if np.all(max(up) < self.tl):
#                     break
#             return up
#
#         def process_output(self, res, txt, numbers):
#             return all(res < self.tl)

# class LMPCQuestion(QuestionGroup):
#     title = "Learning MPC and the car (the final boss)"
#     class PendulumItem(QPrintItem):
#         title = "LMPC lap time"
#         def compute_answer_print(self):
#             from irlc.ex07.lmpc_run import setup_lmpc_controller
#             car, LMPController = setup_lmpc_controller(max_laps=8)
#             stats_, traj_ = train(car, LMPController, num_episodes=1)
#
#         def process_output(self, res, txt, numbers):
#             n = []
#             for t in txt.splitlines():
#                 if not t.startswith("Lap"):
#                     continue
#                 from unitgrade_v1.unitgrade import extract_numbers
#                 nmb = extract_numbers(t)[1]
#                 n.append(nmb)
#             return min(n) < 8.3


class Week07Tests(Report):
    title = "Tests for week 07"
    pack_imports = [irlc]
    individual_imports = []
    questions = [
        (LearningLQRAgentQuestion, 10),             # ok
        (MPCLearningAgentQuestion, 5),              # ok
        (MPCLearningAgentLocalOptimizeQuestion, 10),# ok
        (MPCLocalAgentExactDynamicsQuestion, 5),    # ok
        (MPCLocalLearningLQRAgent, 10),  # ok
        # (MPCLearningAgentLocalOptimizeQuestion, 10),  # ok
        (MPCLearningPendulum, 10),                  # ok
                 ]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week07Tests())

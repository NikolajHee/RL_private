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

class PacmanHardcoded(UTestCase):
    """ Test the hardcoded pacman agent """
    def test_pacman(self):
        env = GymPacmanEnvironment(layout_str=layout)
        agent = GoAroundAgent(env)
        stats, _ = train(env, agent, num_episodes=1)
        self.assertEqual(stats[0]['Length'] < 100, True)


class ChessTournament(UTestCase):
    def test_chess(self):
        """ Test the correct result in the little chess-tournament """
        from irlc.ex01.chess import main
        with Capturing2() as c:
            main()
        # Extract the numbers from the console output.
        print("Numbers extracted from console output was")
        print(c.numbers)
        self.assertLinf(c.numbers[-2], 26/33, tol=0.05)

class InventoryEnvironmentTest(UTestCase):
    def test_environment(self):
        env = InventoryEnvironment()
        # agent = RandomAgent(env)
        stats, _ = train(env, Agent(env), num_episodes=500, verbose=False)
        avg_reward = np.mean([stat['Accumulated Reward'] for stat in stats])
        self.assertLinf(avg_reward, tol=0.5)

    def test_random_agent(self):
        env = InventoryEnvironment()
        stats, _ = train(env, RandomAgent(env), num_episodes=500, verbose=False)
        avg_reward = np.mean([stat['Accumulated Reward'] for stat in stats])
        self.assertLinf(avg_reward, tol=0.5)

    def test_simplified_train(self):
        env = InventoryEnvironment()
        agent = Agent(env)
        avg_reward_simplified_train = np.mean([simplified_train(env, agent) for i in range(1000)])
        self.assertLinf(avg_reward_simplified_train, tol=0.5)

class FrozenLakeTest(UTestCase):
    def test_frozen_lake(self):
        env = gym.make("FrozenLake-v1")
        agent = FrozenAgentDownRight(env)
        s = env.reset()
        for k in range(10):
            self.assertEqual(agent.pi(s, k), DOWN if k % 2 == 0 else RIGHT)


class Week01Tests(Report): #240 total.
    title = "Tests for week 01"
    pack_imports = [irlc]
    individual_imports = []
    questions = [
        (PacmanHardcoded, 10),
        (InventoryEnvironmentTest, 10),
        (FrozenLakeTest, 10),
        (ChessTournament, 10),      # Week 1: Everything
                 ]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week01Tests())

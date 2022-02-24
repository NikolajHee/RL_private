# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import Report, UTestCase, cache
import numpy as np
from irlc import train
import irlc.ex09.envs
import gym
from irlc.tests.tests_week08 import train_recording


class MCAgentQuestion(UTestCase):
    """ Test of MC agent """
    def get_env_agent(self):
        from irlc.ex10.mc_agent import MCAgent
        import gym
        env = gym.make("SmallGridworld-v0")
        from gym.wrappers import TimeLimit
        env = TimeLimit(env, max_episode_steps=1000)
        gamma = .8
        agent = MCAgent(env, gamma=gamma, first_visit=True)
        return env, agent

    @cache
    def compute_trajectories(self):
        env, agent = self.get_env_agent()
        _, trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
        return trajectories, agent.Q.to_dict()

    def test_Q_function(self):
        trajectories, Q = self.compute_trajectories()
        env, agent = self.get_env_agent()
        train_recording(env, agent, trajectories)
        Qc = []
        Qe = []
        for s, qa in Q.items():
            for a,q in qa.items():
                Qe.append(q)
                Qc.append(agent.Q[s,a])

        self.assertL2(Qe, Qc, tol=1e-5)


class BlackjackQuestion(UTestCase):
    """ MC policy evaluation agent and Blacjack """
    def test_blackjack_mc(self):
        env = gym.make("Blackjack-v0")
        episodes = 50000
        from irlc.ex10.mc_evaluate import MCEvaluationAgent
        from irlc.ex10.mc_evaluate_blackjack import get_by_ace, to_matrix, policy20
        agent = MCEvaluationAgent(env, policy=policy20, gamma=1)
        train(env, agent, num_episodes=episodes)
        w = get_by_ace(agent.v, ace=True)
        X, Y, Z = to_matrix(w)
        self.assertL2(Z, tol=2)


class TD0Question(UTestCase):
    """ Test of TD(0) evaluation agent """
    gamma = 0.8

    def get_env_agent(self):
        from irlc.ex10.td0_evaluate import TD0ValueAgent
        env = gym.make("SmallGridworld-v0")
        from gym.wrappers import TimeLimit
        env = TimeLimit(env, max_episode_steps=1000)
        agent = TD0ValueAgent(env, gamma=self.gamma)
        return env, agent

    @cache
    def compute_trajectories(self):
        env, agent = self.get_env_agent()
        _, trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
        return trajectories, agent.v

    def test_value_function(self):
        trajectories, v = self.compute_trajectories()
        env, agent = self.get_env_agent()
        train_recording(env, agent, trajectories)
        Qc = []
        Qe = []
        for s, value in v.items():
            Qe.append(value)
            Qc.append(agent.v[s])

        self.assertL2(Qe, Qc, tol=1e-5)

class MCEvaluationQuestion(TD0Question):
    """ Test of MC evaluation agent """
    def get_env_agent(self):
        from irlc.ex10.mc_evaluate import MCEvaluationAgent
        import gym
        env = gym.make("SmallGridworld-v0")
        from gym.wrappers import TimeLimit
        env = TimeLimit(env, max_episode_steps=1000)
        gamma = .8
        agent = MCEvaluationAgent(env, gamma=gamma, first_visit=True)
        return env, agent


class Week10Tests(Report):
    title = "Tests for week 10"
    pack_imports = [irlc]
    individual_imports = []
    questions = [(MCAgentQuestion, 10),
                (MCEvaluationQuestion, 10),
                (BlackjackQuestion,5),
                 (TD0Question, 10),
                 ]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week10Tests())

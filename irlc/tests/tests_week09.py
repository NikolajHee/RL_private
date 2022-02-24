# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, Report

import numpy as np
from irlc import train
import irlc.ex09.envs
import gym

def train_recording(env, agent, trajectories):
    for t in trajectories:
        env.reset()
        for k in range(len(t.action)):
            s = t.state[k]
            r = t.reward[k]
            a = t.action[k]
            sp = t.state[k+1]
            agent.pi(s,k)
            agent.train(s, a, r, sp, done=k == len(t.action)-1)

# Now, the ValueIteartionAgent group
# class GridworldDPItem(QPrintItem):
#     testfun = QPrintItem.assertL2
#     title = "Small Gridworld"
#     tol = 1e-3
#
#     def get_value_function(self):
#         from irlc.ex09.small_gridworld import SmallGridworldMDP
#         from irlc.ex09.policy_evaluation import policy_evaluation
#         env = SmallGridworldMDP()
#         pi0 = {s: {a: 1 / len(env.A(s)) for a in env.A(s)} for s in env.nonterminal_states}
#         V = policy_evaluation(pi0, env, gamma=.83)
#         return V, env
#
#     def compute_answer_print(self):
#         V, env = self.get_value_function()
#         return np.asarray( [V[s] for s in env.states] )
#
#     def process_output(self, res, txt, numbers):
#         return res

class ValueFunctionTest(UTestCase):
    def check_value_function(self, env, V):
        self.assertL2(np.asarray([V[s] for s in env.states]), tol=1e-3)

from irlc.ex09.small_gridworld import SmallGridworldMDP
from irlc.ex09.policy_iteration import policy_iteration
from irlc.ex09.value_iteration import value_iteration

class PolicyIterationQuestion(ValueFunctionTest):
    """ Iterative Policy iteration """
    def test_policy_iteration(self):
        env = SmallGridworldMDP()
        pi, v = policy_iteration(env, gamma=0.91)
        self.check_value_function(env, v)

class PolicyEvaluationQuestion(ValueFunctionTest):
    """ Iterative value iteration """
    def test_value_iteration(self):
        env = SmallGridworldMDP()
        # from i
        pi, v = value_iteration(env, gamma=0.91)
        self.check_value_function(env, v)


class TestGambler(ValueFunctionTest):
    """ Gambler's problem """
    def test_gambler_value_function(self):
        # from irlc.ex09.small_gridworld import SmallGridworldMDP, plot_value_function
        # from irlc.ex09.policy_iteration import policy_iteration
        # from irlc.ex09.value_iteration import value_iteration
        from irlc.ex09.gambler import GamblerEnv
        env = GamblerEnv()
        pi, v = value_iteration(env, gamma=0.91)
        self.check_value_function(env, v)

class JackQuestion(ValueFunctionTest):
    """ Gambler's problem """
    def test_jacks_rental_value_function(self):
        # from irlc.ex09.small_gridworld import SmallGridworldMDP, plot_value_function
        # from irlc.ex09.policy_iteration import policy_iteration
        # from irlc.ex09.value_iteration import value_iteration
        # from irlc.ex09.gambler import GamblerEnv
        from irlc.ex09.jacks_car_rental import JackRentalMDP
        max_cars = 5
        env = JackRentalMDP(max_cars=max_cars, verbose=True)
        pi, V = value_iteration(env, gamma=.9, theta=1e-3, max_iters=1000, verbose=True)
        self.check_value_function(env, V)

# class JackQuestion(QuestionGroup):
#     title = "Jacks car rental problem"
#
#     class JackItem(GridworldDPItem):
#         title = "Value function test"
#         max_cars = 5
#         tol = 0.01
#
#         def get_value_function(self):
#             from irlc.ex09.value_iteration import value_iteration
#             from irlc.ex09.jacks_car_rental import JackRentalMDP
#             env = JackRentalMDP(max_cars=self.max_cars, verbose=True)
#             pi, V = value_iteration(env, gamma=.9, theta=1e-3, max_iters=1000, verbose=True)
#             return V, env


        # return v, env
    # pass
# class DynamicalProgrammingGroup(QuestionGroup):
#     title = "Dynamical Programming test"
#
#     class PolicyEvaluationItem(GridworldDPItem):
#         title = "Iterative Policy evaluation"
#
#
#
#     class PolicyIterationItem(GridworldDPItem):
#         title = "policy iteration"
#         def get_value_function(self):
#             from irlc.ex09.small_gridworld import SmallGridworldMDP
#             from irlc.ex09.policy_iteration import policy_iteration
#             env = SmallGridworldMDP()
#             pi, v = policy_iteration(env, gamma=0.91)
#             return v, env
#     class ValueIteartionItem(GridworldDPItem):
#         title = "value iteration"
#
#         def get_value_function(self):
#             from irlc.ex09.value_iteration import value_iteration
#             from irlc.ex09.small_gridworld import SmallGridworldMDP
#             env = SmallGridworldMDP()
#             policy, v = value_iteration(env, gamma=0.92, theta=1e-6)
#             return v, env

# class GamlerQuestion(QuestionGroup):
#     title = "Gamblers problem"
#     class GamlerItem(GridworldDPItem):
#         title = "Value-function test"
#         def get_value_function(self):
#             # from irlc.ex09.small_gridworld import SmallGridworldMDP, plot_value_function
#             # from irlc.ex09.policy_iteration import policy_iteration
#             from irlc.ex09.value_iteration import value_iteration
#             from irlc.ex09.gambler import GamblerEnv
#             env = GamblerEnv()
#             pi, v = value_iteration(env, gamma=0.91)
#             return v, env

# class JackQuestion(QuestionGroup):
#     title ="Jacks car rental problem"
#     class JackItem(GridworldDPItem):
#         title = "Value function test"
#         max_cars = 5
#         tol = 0.01
#         def get_value_function(self):
#             from irlc.ex09.value_iteration import value_iteration
#             from irlc.ex09.jacks_car_rental import JackRentalMDP
#             env = JackRentalMDP(max_cars=self.max_cars, verbose=True)
#             pi, V = value_iteration(env, gamma=.9, theta=1e-3, max_iters=1000, verbose=True)
#             return V, env

class DPAgentRLQuestion(UTestCase):
    """ Value-iteration agent test """

    def test_sutton_gridworld(self):
        tol = 1e-2
        from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment
        env = SuttonCornerGridEnvironment(living_reward=-1)
        from irlc.ex09.value_iteration_agent import ValueIterationAgent
        agent = ValueIterationAgent(env, mdp=env.mdp)
        stats, _ = train(env, agent, num_episodes=1000)
        self.assertL2(np.mean([s['Accumulated Reward'] for s in stats]), tol=tol)

    def test_bookgrid_gridworld(self):
        tol = 1e-2
        from irlc.gridworld.gridworld_environments import BookGridEnvironment
        env = BookGridEnvironment(living_reward=-1)
        from irlc.ex09.value_iteration_agent import ValueIterationAgent
        agent = ValueIterationAgent(env, mdp=env.mdp)
        stats, _ = train(env, agent, num_episodes=1000)
        self.assertL2(np.mean([s['Accumulated Reward'] for s in stats]), tol=tol)


    #
    #
    #     pass
    # class ValueAgentItem(GridworldDPItem):
    #     title = "Evaluation on Suttons small gridworld"
    #     tol = 1e-2
    #     def get_env(self):
    #         from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment
    #         return SuttonCornerGridEnvironment(living_reward=-1)
    #
    #     def compute_answer_print(self):
    #         env = self.get_env()
    #         from irlc.ex09.value_iteration_agent import ValueIterationAgent
    #         agent = ValueIterationAgent(env, mdp=env.mdp)
    #         # env = VideoMonitor(env, agent=agent, agent_monitor_keys=('v',))
    #         stats, _ = train(env, agent, num_episodes=1000)
    #         return np.mean( [s['Accumulated Reward'] for s in stats])
    #
    #     def process_output(self, res, txt, numbers):
    #         return res

    # class BookItem(ValueAgentItem):
    #     title = "Evaluation on alternative gridworld (Bookgrid)"
    #     def get_env(self):
    #         from irlc.gridworld.gridworld_environments import BookGridEnvironment
    #         return BookGridEnvironment(living_reward=-0.6)

# class DPAgentRLQuestion(QuestionGroup):
#     title = "Value-iteration agent test"
#     class ValueAgentItem(GridworldDPItem):
#         title = "Evaluation on Suttons small gridworld"
#         tol = 1e-2
#         def get_env(self):
#             from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment
#             return SuttonCornerGridEnvironment(living_reward=-1)
#
#         def compute_answer_print(self):
#             env = self.get_env()
#             from irlc.ex09.value_iteration_agent import ValueIterationAgent
#             agent = ValueIterationAgent(env, mdp=env.mdp)
#             # env = VideoMonitor(env, agent=agent, agent_monitor_keys=('v',))
#             stats, _ = train(env, agent, num_episodes=1000)
#             return np.mean( [s['Accumulated Reward'] for s in stats])
#
#         def process_output(self, res, txt, numbers):
#             return res
#
#     class BookItem(ValueAgentItem):
#         title = "Evaluation on alternative gridworld (Bookgrid)"
#         def get_env(self):
#             from irlc.gridworld.gridworld_environments import BookGridEnvironment
#             return BookGridEnvironment(living_reward=-0.6)

class Week09Tests(Report):
    title = "Tests for week 09"
    pack_imports = [irlc]
    individual_imports = []
    questions = [ (PolicyIterationQuestion, 10),
        # (ValueFunctionTest, 20),
                    (PolicyEvaluationQuestion, 10),
                  (TestGambler, 10),
                    (JackQuestion, 10),
                    (DPAgentRLQuestion, 5),
                  ]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week09Tests())

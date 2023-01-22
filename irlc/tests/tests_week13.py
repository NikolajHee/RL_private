# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, Report
import numpy as np
from irlc import train
import irlc.ex09.envs
from irlc.tests.tests_week11 import TabularAgentStub

class DoubleQQuestion(TabularAgentStub):
    """ Double Q learning """
    def test_accumulated_reward(self):
        env, agent = self.get_env_agent()
        stats, _ = train(env, agent, num_episodes=5000)
        s, info = env.reset()
        actions, qs = agent.Q1.get_Qs(s, info)
        self.assertL2(qs, tol=10)
        self.assertL2(np.mean([s['Accumulated Reward'] for s in stats]), tol=self.tol)
        return stats

    def get_env_agent(self):
        from irlc.ex13.tabular_double_q import TabularDoubleQ
        agent = TabularDoubleQ(self.get_env(), gamma=self.gamma)
        return agent.env, agent


class DynaQQuestion(TabularAgentStub):
    """ Dyna Q learning """
    # class DynaQReturnItem(SarsaReturnTypeItem):
    def get_env_agent(self):
        from irlc.ex13.dyna_q import DynaQ
        agent = DynaQ(self.get_env(), gamma=self.gamma)
        return agent.env, agent

    def test_accumulated_reward(self):
        self.chk_accumulated_reward()

    # class DynaQItem(SarsaTypeQItem):
    #     title = "Dyna Q action distribution"

# class DoubleQQuestion(QuestionGroup):
#     title = "Double Q learning"
#     class DQReturnItem(SarsaReturnTypeItem):
#         def get_env_agent(self):
#             from irlc.ex13.tabular_double_q import TabularDoubleQ
#             agent = TabularDoubleQ(self.get_env(), gamma=self.gamma)
#             return agent.env, agent
#
#     class DoubleQItem(SarsaTypeQItem):
#         tol = 1
#         def compute_answer_print(self):
#             s = self.question.env.reset()
#             actions, qs = self.question.agent.Q1.get_Qs(s)
#             return qs
#         title = "Double Q action distribution"
#
# class DynaQQuestion(QuestionGroup):
#     title = "Dyna Q learning"
#     class DynaQReturnItem(SarsaReturnTypeItem):
#         def get_env_agent(self):
#             from irlc.ex13.dyna_q import DynaQ
#             agent = DynaQ(self.get_env(), gamma=self.gamma)
#             return agent.env, agent
#
#     class DynaQItem(SarsaTypeQItem):
#         title = "Dyna Q action distribution"

class Week13Tests(Report):
    title = "Tests for week 13"
    pack_imports = [irlc]
    individual_imports = []
    questions =  [(DoubleQQuestion, 10),
                 (DynaQQuestion, 10)
                 ]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week13Tests())

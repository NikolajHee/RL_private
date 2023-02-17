# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, Report
import irlc
from irlc import train
import numpy as np

matrices = ['L', 'l', 'V', 'v', 'vc']
class LQRQuestion(UTestCase):
    title = "LQR, full check of implementation"

    @classmethod
    def setUpClass(cls):
        # def init(self):
        from irlc.ex06.dlqr_check import check_LQR
        (cls.L, cls.l), (cls.V, cls.v, cls.vc) = check_LQR()
        # self.M = list(zip(matrices, [L, l, V, v, vc]))

    def chk_item(self, m_list):
        self.assertIsInstance(m_list, list)
        self.assertEqualC(len(m_list))
        for m in m_list:
            self.assertIsInstance(m, np.ndarray)
            self.assertEqualC(m.shape)
            self.assertL2(m, tol=1e-6)

    def test_L(self):
        self.chk_item(self.__class__.L)

    def test_l(self):
        self.chk_item(self.__class__.l)

    def test_V(self):
        self.chk_item(self.__class__.V)

    def test_v(self):
        self.chk_item(self.__class__.v)

    def test_vc(self):
        vc = self.__class__.vc
        self.assertIsInstance(vc, list)
        for d in vc:
            self.assertL2(d, tol=1e-6)

        self.chk_item(self.__class__.l)



class BoingQuestion(UTestCase):
    """ Boing flight control with LQR """

    def test_boing(self):
        from irlc.ex06.boing_lqr import boing_simulation
        stats, trajectories, env = boing_simulation()
        self.assertL2(trajectories[-1].state, tol=1e-6)


class RendevouzItem(UTestCase):
    def test_rendevouz_without_linesearch(self):
        """ Rendevouz with iLQR (no linesearch) """
        from irlc.ex06.ilqr_rendovouz_basic import solve_rendovouz
        (xs, us, J_hist, l, L), env = solve_rendovouz(use_linesearch=False)
        # print(J_hist[-1])
        self.assertL2(xs[-1], tol=1e-2)

    def test_rendevouz_with_linesearch(self):
        """ Rendevouz with iLQR (with linesearch) """
        from irlc.ex06.ilqr_rendovouz_basic import solve_rendovouz
        (xs, us, J_hist, l, L), env = solve_rendovouz(use_linesearch=True)
        # print(J_hist[-1])
        self.assertL2(xs[-1], tol=1e-2)
        # return l, L, xs



class ILQRAgentQuestion(UTestCase):
    """ iLQR Agent on Rendevouz """
    def test_ilqr_agent(self):
        from irlc.ex06.ilqr_agent import solve_rendevouz
        stats, trajectories, agent = solve_rendevouz()
        self.assertL2(trajectories[-1].state[-1], tol=1e-2)


class ILQRPendulumQuestion(UTestCase):
    """ iLQR Agent on Pendulum """

    def test_ilqr_agent_pendulum(self):
        from irlc.ex06.ilqr_pendulum_agent import Tmax, N
        from irlc.ex04.model_pendulum import GymSinCosPendulumEnvironment
        from irlc.ex06.ilqr_agent import ILQRAgent
        dt = Tmax / N
        env = GymSinCosPendulumEnvironment(dt, Tmax=Tmax, supersample_trajectory=True)
        agent = ILQRAgent(env, env.discrete_model, N=N, ilqr_iterations=200, use_linesearch=True)
        stats, trajectories = train(env, agent, num_episodes=1, return_trajectory=True)
        state = trajectories[-1].state[-1]
        self.assertL2(state, tol=2e-2)


class Week06Tests(Report):
    title = "Tests for week 06"
    pack_imports = [irlc]
    individual_imports = []
    questions = [
        (LQRQuestion, 10),                          # ok ILQR
        (BoingQuestion, 10),                        # ok
        (RendevouzItem, 10),                        # ok
        (ILQRAgentQuestion,10),                     # ok
        (ILQRPendulumQuestion, 10),                 # ok
                 ]
if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week06Tests())

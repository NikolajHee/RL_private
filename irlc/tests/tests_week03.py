# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import Report
import irlc
from unitgrade import UTestCase
from irlc.ex03.dp_forward import dp_forward
from irlc.ex03.search_problem import GraphSP, DP2SP
from irlc.ex03.search_problem import EnsureTerminalSelfTransitionsWrapper
from irlc.ex02.graph_traversal import SmallGraphDP

class ForwardDPTests(UTestCase):
    def test_partA(self):
        t = 5
        s = 2
        sp = GraphSP(start=s, goal=t)
        N = len(set([i for edge in sp.G for i in edge])) - 1
        J_sp, pi_sp, path = dp_forward(sp, N)

        print(N)
        print(path)
        print("PART A: Search cost was", J_sp[-1][t])
        self.assertEqualC(J_sp[-1][t])

    def test_partB(self):
        s, t = 2, 5
        sp = GraphSP(start=s, goal=t)
        N = len(set([i for edge in sp.G for i in edge])) - 1
        sp_wrapped = EnsureTerminalSelfTransitionsWrapper(sp)
        J_wrapped, pi_wrapped, path_wrapped = dp_forward(sp_wrapped, N)
        print("PART B: Search cost was", J_wrapped[-1][t])
        self.assertEqualC(J_wrapped[-1][t])

    def test_dp2sp(self):
        s, t = 2, 5
        env = SmallGraphDP(t=t)
        sp_env = DP2SP(env, initial_state=s)
        self.assertEqualC(sp_env.available_transitions((s,0)))

    def test_partC(self):
        s, t = 2, 5
        env = SmallGraphDP(t=t)
        sp_env = DP2SP(env, initial_state=s)
        J2, pi2, path = dp_forward(sp_env, env.N)
        self.assertEqualC(J2[-1][sp_env.terminal_state])

class Travelman(UTestCase):
    def test_travelman(self):
        from irlc.ex03.travelman import main, TravelingSalesman
        tm = TravelingSalesman()
        s = ("A",)
        tm_sp = DP2SP(tm, s)
        J, actions, path = dp_forward(tm_sp, N=tm.N)
        self.assertEqualC(J[-1][tm_sp.terminal_state])

class Queens(UTestCase):
    def test_travelman(self):
        from irlc.ex03.queens import QueensDP
        from irlc.ex03.search_problem import DP2SP
        N = 4
        q = QueensDP(N)
        s = ()  # first state is the empty chessboard
        q_sp = DP2SP(q, initial_state=s)
        J, actions, path = dp_forward(q_sp, N)
        board = path[-2][0]
        self.assertEqual(QueensDP.valid_pos_(q, board[:-1], board[-1]), True)

from irlc.ex03.pacman_search import PacmanSearchProblem, layout1, layout2
from irlc.pacman.pacman_environment import GymPacmanEnvironment

class PacmanSearch(UTestCase):
    def test_pacman_search_problem(self):


        # PacmanSearchProblem
        # N = 20  # Plan over a horizon of N steps.
        env1 = GymPacmanEnvironment(layout_str=layout1)  # Create environment
        problem1 = PacmanSearchProblem(env1)  # Transform into a search problem
        self.assertEqualC(problem1.available_transitions(env1.reset()))
    def test_costs(self):
        N = 20  # Plan over a horizon of N steps.
        env1 = GymPacmanEnvironment(layout_str=layout1)
        problem1 = EnsureTerminalSelfTransitionsWrapper(PacmanSearchProblem(env1))  # Transform into a search problem

        env2 = GymPacmanEnvironment(layout_str=layout2)
        problem2 = EnsureTerminalSelfTransitionsWrapper(PacmanSearchProblem(env2))  # Transform into a search problem

        J1, _, _ = dp_forward(problem1, N)  # Compute optimal trajectory in layout 1
        J2, _, _ = dp_forward(problem2, N)  # Compute optimal trajectory in layout 2
        optimal_cost_1 = min([cost for s, cost in J1[-1].items() if problem1.is_terminal(s)])
        optimal_cost_2 = min([cost for s, cost in J2[-1].items() if problem1.is_terminal(s)])

        self.assertLinf(optimal_cost_1, tol=1e-8)
        self.assertLinf(optimal_cost_2, tol=1e-8)


# class Travelman(QPrintItem):
#     def compute_answer_print(self):
#         from irlc.ex03.travelman import main
#         main()
#
#     def process_output(self, res, txt, numbers):
#         return numbers[:2]
#
#
# class NQueens(QPrintItem):
#     def compute_answer_print(self):
#         from irlc.ex03.queens import QueensDP
#         from irlc.ex03.search_problem import DP2SP
#         N = 4
#         q = QueensDP(N)
#         s = ()  # first state is the empty chessboard
#         q_sp = DP2SP(q, initial_state=s)
#         J, actions, path = dp_forward(q_sp, N)
#         board = path[-2][0]
#         return QueensDP.valid_pos_(q, board[:-1], board[-1] )
#
#     def process_output(self, res, txt, numbers):
#         return res

class Week03Tests(Report):
    title = "Tests for week 03"
    pack_imports = [irlc]
    individual_imports = []
    questions = [
        (ForwardDPTests, 10),
        (Travelman, 10),
        (Queens, 10),
        (PacmanSearch, 10),
         ]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week03Tests())

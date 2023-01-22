# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import Report
from irlc.ex05.direct_agent import train_direct_agent
from unitgrade import UTestCase
import irlc
from irlc.ex05.direct import run_direct_small_problem


class DirectMethods(UTestCase):
    title = "Direct methods z, z0, z_lb/z_ub definitions+"

    @classmethod
    def setUpClass(cls) -> None:
        env, solution = run_direct_small_problem()
        cls.solution = solution[-1]


    def test_z_variable_vector(self):
        self.assertEqualC(str(DirectMethods.solution['inputs']['z']))

    def test_z0_initial_state(self):
        self.assertL2(DirectMethods.solution['inputs']['z0'], tol=1e-6)

    def test_zU_upper_bound(self):
        self.assertL2(DirectMethods.solution['inputs']['z_ub'], tol=1e-6)

    def test_zL_lower_bound(self):
        self.assertL2(DirectMethods.solution['inputs']['z_lb'], tol=1e-6)


class DirectAgentPendulum(UTestCase):
    """ Direct agent: Test of pendulum environment """
    def test_pendulum(self):
        stats = train_direct_agent(animate=False)
        return self.assertL2(stats[0]['Accumulated Reward'], tol=0.03)

class DirectSolverQuestion(UTestCase):
    """ Test the Direct solver on the Pendulum using run_direct_small_problem() """
    @classmethod
    def setUpClass(cls):
        cls.solution = cls.compute_solution()

    @classmethod
    def compute_solution(cls):
        from irlc.ex05.direct import run_direct_small_problem
        env, solution = run_direct_small_problem()
        return solution
        # cls.solution = solution

    def test_solver_success(self):
        self.assertTrue(self.__class__.solution[-1]['solver']['success'])

    def test_solver_fun(self):
        self.assertL2(self.__class__.solution[-1]['solver']['fun'], tol=0.01)

    def test_constraint_violation(self):
        self.assertL2(self.__class__.solution[-1]['eqC_val'], tol=0.01)

    # def test_constraint_violation(self):
    #     self.assertL2(self.__class__.question.solutions[-1]['solver']['eqC_val'], tol=0.01)


# class SuccessItem_(QPrintItem):
#     def compute_answer_print(self):
#         return self.question.solutions[-1]['solver']['success']
#
# class CostItem_(QPrintItem):
#     tol = 0.01
#     def compute_answer_print(self):
#         return self.question.solutions[-1]['solver']['fun']
#
#     def process_output(self, res, txt, numbers):
#         return res
#
# class ConstraintVioloationItem_(QPrintItem):
#     tol = 0.01
#     def compute_answer_print(self):
#         return self.question.solutions[-1]['eqC_val']
#
#     def process_output(self, res, txt, numbers):
#         return res + 1e-5

# class DirectSolverQuestion_(QuestionGroup):
#     title = "Direct solver on a small problem"
#     def compute_solutions(self):
#         from irlc.ex05.direct import run_direct_small_problem
#         env, solution = run_direct_small_problem()
#         return solution
#
#     def init(self):
#         # super().__init__(*args, **kwargs)
#         with Capturing():
#             self.solutions = self.compute_solutions()

# class SmallDirectProblem(DirectSolverQuestion_):
#     def compute_solutions(self):
#         from irlc.ex05.direct import run_direct_small_problem
#         env, solution = run_direct_small_problem()
#         return solution
#
#     class SuccessItem(SuccessItem_):
#         pass
#     class CostItem_(CostItem_):
#         pass
#     class ConstraintVioloationItem_(ConstraintVioloationItem_):
#         pass

# class PendulumQuestion(DirectSolverQuestion_):
#     title = "Direct solver on the pendulum problem"
#     def compute_solutions(self):
#         from irlc.ex05.direct_pendulum import compute_pendulum_solutions
#         return compute_pendulum_solutions()[1]
#     class SuccessItem(SuccessItem_):
#         pass
#     class CostItem_(CostItem_):
#         pass
#     class ConstraintVioloationItem_(ConstraintVioloationItem_):
#         pass


class PendulumQuestion(DirectSolverQuestion):
    """ Direct solver on the pendulum problem """
    @classmethod
    def compute_solution(cls):
        from irlc.ex05.direct_pendulum import compute_pendulum_solutions
        return compute_pendulum_solutions()[1]


class CartpoleTimeQuestion(DirectSolverQuestion):
    """ Direct solver on the cartpole (minimum time) task """
    @classmethod
    def compute_solution(cls):
        from irlc.ex05.direct_cartpole_time import compute_solutions
        return compute_solutions()[1]


#
# class CartpoleTimeQuestion(DirectSolverQuestion_):
#     title = "Direct solver on the cartpole (minimum time) task"
#     def compute_solutions(self):
#         from irlc.ex05.direct_cartpole_time import compute_solutions
#         return compute_solutions()[1]
#     class SuccessItem(SuccessItem_):
#         pass
#     class CostItem_(CostItem_):
#         pass
#     class ConstraintVioloationItem_(ConstraintVioloationItem_):
#         pass

# class CartpoleCostQuestion(DirectSolverQuestion_):
#     title = "Direct solver on the cartpole (kelly) task"
#     def compute_solutions(self):
#         from irlc.ex05.direct_cartpole_kelly import compute_solutions
#         return compute_solutions()[1]
#     class SuccessItem(SuccessItem_):
#         pass
#     class CostItem_(CostItem_):
#         pass
#     class ConstraintVioloationItem_(ConstraintVioloationItem_):
#         pass


class CartpoleCostQuestion(DirectSolverQuestion):
    """ Direct solver on the cartpole (kelly) task """

    @classmethod
    def compute_solution(cls):
        from irlc.ex05.direct_cartpole_kelly import compute_solutions
        return compute_solutions()[1]


# class BrachistochroneQuestion(DirectSolverQuestion_):
#     title = "Brachistochrone (unconstrained)"
#     def compute_solutions(self):
#         from irlc.ex05.direct_brachistochrone import compute_unconstrained_solutions
#         return compute_unconstrained_solutions()[1]
#     class SuccessItem(SuccessItem_):
#         pass
#     class CostItem_(CostItem_):
#         pass
#     class ConstraintVioloationItem_(ConstraintVioloationItem_):
#         pass

# class BrachistochroneConstrainedQuestion(DirectSolverQuestion_):
#     title = "Brachistochrone (constrained)"
#     def compute_solutions(self):
#         from irlc.ex05.direct_brachistochrone import compute_constrained_solutions
#         return compute_constrained_solutions()[1]
#     class SuccessItem(SuccessItem_):
#         pass
#     class CostItem_(CostItem_):
#         pass
#     class ConstraintVioloationItem_(ConstraintVioloationItem_):
#         pass

class BrachistochroneQuestion(DirectSolverQuestion):
    """ Brachistochrone (unconstrained) """

    @classmethod
    def compute_solution(cls):
        from irlc.ex05.direct_brachistochrone import compute_constrained_solutions
        return compute_constrained_solutions()[1]

class BrachistochroneConstrainedQuestion(DirectSolverQuestion):
    """ Brachistochrone (constrained) """

    @classmethod
    def compute_solution(cls):
        from irlc.ex05.direct_brachistochrone import compute_constrained_solutions
        return compute_constrained_solutions()[1]

class Week05Tests(Report):
    title = "Tests for week 05"
    pack_imports = [irlc]
    individual_imports = []
    questions = [
        (DirectMethods, 10),                        # ok
        (DirectSolverQuestion, 10),                   # ok
        (PendulumQuestion, 5),                      # ok
        (DirectAgentPendulum, 10),                  # ok
        (CartpoleTimeQuestion, 5),                  # ok
        (CartpoleCostQuestion, 5),                  # ok
        (BrachistochroneQuestion, 5),               # ok
        (BrachistochroneConstrainedQuestion, 10),   # ok
                 ]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week05Tests())

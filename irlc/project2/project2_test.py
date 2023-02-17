# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, Report
import irlc
import numpy as np

class YodaProblem1(UTestCase):
    """ Test the get_A_B() function (Section 1, Problem 1) """
    def test_A_B(self):
        from irlc.project2.yoda_part1 import get_A_B
        for g in [9.82, 5.1]:
            for L in [0.2, 0.5, 1.1]:
                A,B = get_A_B(g,L)
                # To get the expected output of a test (in the cases where it is not specified manually),
                # simply use the line self.get_expected_test_value() *right before* running the test-function itself.
                print("The expected value is", self.get_expected_test_value())
                # If the code does not work, you need to upgrade unitgrade to the most recent version:
                # pip install unitgrade --upgrade --no-cache
                self.assertLinf(A)
                self.assertLinf(B)

class YodaProblem3(UTestCase):
    """ Stopping pendulum using LQR (Section 1, Problem 3) """
    def test_cost_discrete(self):
        """ Test the discrete cost c_k """
        from irlc.project2.yoda_part1 import cost_discrete
        x = np.zeros( (2,))
        u = np.zeros((1,))
        self.assertL2( cost_discrete(x, u), 0) # zero state/action is zero cost

        x = np.ones((2,))
        u = np.ones((1,))
        self.assertL2(cost_discrete(x, u*0))  # test Q_k-part
        self.assertL2(cost_discrete(x*0, u))  # test R_k-part
        self.assertL2(cost_discrete(np.asarray([1.1, -3]), np.asarray([-3])))  # test for random values

    def test_L_and_l(self):
        """ Test the control-matrices L_0 and l_0. """
        from irlc.project2.yoda_part1 import part1
        for L in [0.3, 0.45, 1.3]: # Test different string lengths.
            L, l, action = part1(L)
            self.assertLinf(L, tol=1e-4)
            self.assertLinf(l, tol=1e-4)

    def test_actions(self):
        from irlc.project2.yoda_part1 import part1
        for L in [0.3, 0.65, 1.2]:  # Test different string lengths.
            L, l, actions = part1(L)
            actions = actions.reshape((-1,))
            self.assertEqualC(actions.size) # Test the number of actions
            self.assertLinf(actions, tol=1e-5) # test the value.

class YodaProblem4(UTestCase):
    """ Stopping pendulum using PID (Section 1, Problem 4) """
    def test_PID(self):
        from irlc.project2.yoda_part1 import part2
        for L in [0.3, 0.45, 1.3]:  # Test different string lengths.
            Kp, Ki, Kd, x_star, actions = part2(L)
            self.assertLinf(x_star, tol=1e-4)
            self.assertLinf(Kp, tol=1e-4)
            self.assertLinf(Kd, tol=1e-4)
            self.assertLinf(Ki, tol=1e-4)

    def test_actions(self):
        from irlc.project2.yoda_part1 import part2
        for L in [0.3, 0.65, 1.2]:  # Test different string lengths.
            Kp, Ki, Kd, x_star, actions = part2(L)
            actions = actions.reshape((-1,))
            self.assertEqualC(actions.size)  # Test the number of actions
            self.assertLinf(actions, tol=1e-5)  # test the value.


class YodaProblem6(UTestCase):
    """ Yoda part 2: Problem 6 """
    def test_A0(self):
        from irlc.project2.yoda_part2 import A_ei, A_euler
        for g in [9.2, 10]:
            for L in [0.2, 0.4]:
                for Delta in [0.1, 0.2]:
                    self.assertLinf(A_euler(g, L, Delta)) # Test Euler discretization
                    self.assertLinf(A_ei(g, L, Delta))    # Test exponential discretization


class YodaProblem7(UTestCase):
    """ Yoda part 2: Problem 7 """
    def test_M(self):
        from irlc.project2.yoda_part2 import M_ei, M_euler
        for g in [9.2, 10]:
            for L in [0.2, 0.4]:
                for Delta in [0.1, 0.2]:
                    for N in [3, 5]:
                        self.assertLinf(M_ei(g, L, Delta, N)) # Test Euler discretization
                        self.assertLinf(M_euler(g, L, Delta, N))    # Test exponential discretization


class YodaProblem11(UTestCase):
    """ Yoda part 2: Bound using Euler discretization Problem 10 """
    def test_xN_euler_bound(self):
        from irlc.project2.yoda_part2 import xN_bound_euler
        for g in [9.2, 10]:
            for L in [0.2, 0.4]:
                for Delta in [0.1, 0.2]:
                    for N in [3, 5]:
                        self.assertLinf(xN_bound_euler(g, L, Delta, N))

class YodaProblem12(UTestCase):
    """ Yoda part 2: Bound using exponential discretization Problem 11 """
    def test_xN_euler_bound(self):
        from irlc.project2.yoda_part2 import xN_bound_ei
        for g in [9.2, 10]:
            for L in [0.2, 0.4]:
                for Delta in [0.1, 0.2]:
                    for N in [3, 5]:
                        self.assertLinf(xN_bound_ei(g, L, Delta, N))


class R2D2Problem15(UTestCase):
    """ Check the linearization and discretization code in Problem 13 and Problem 14"""
    def test_f_euler_zeros(self):
        # Test in a simple case:
        x = np.zeros((3,))
        u = np.asarray([1,0])
        from irlc.project2.r2d2 import f_euler
        self.assertLinf(f_euler(x, u, Delta=0.05))
        self.assertLinf(f_euler(x, u, Delta=0.1))

    def test_f_euler(self):
        np.random.seed(42)
        for _ in range(4):
            x = np.random.randn(3)
            u = np.random.randn(2)
            from irlc.project2.r2d2 import f_euler
            self.assertLinf(f_euler(x, u, Delta=0.05))
            self.assertLinf(f_euler(x, u, Delta=0.1))

    def checklin(self, x_bar, u_bar):
        from irlc.project2.r2d2 import linearize
        A, B, d = linearize(x_bar, u_bar, Delta=0.05)
        self.assertLinf(A)
        self.assertLinf(B)
        self.assertLinf(d)

    def test_linearization1(self):
        x_bar = np.asarray([0, 0, 0])
        u_bar = np.asarray([1, 0])
        self.checklin(x_bar, u_bar)

    def test_linearization2(self):
        x_bar = np.asarray([0, 0, 0.24])
        u_bar = np.asarray([1, 0])
        self.checklin(x_bar, u_bar)

    def test_linearization3(self):
        np.random.seed(42)
        for _ in range(10):
            x_bar = np.random.randn(3)
            u_bar = np.asarray([1, 0])
            self.checklin(x_bar, u_bar)

class R2D2Direct(UTestCase):
    """ Problem 16: R2D2 and direct methods """

    def chk_direct(self, x_target):
        from irlc.project2.r2d2 import drive_to_direct
        states = drive_to_direct(x_target=x_target, plot=False)
        self.assertIsInstance(states, np.ndarray)  # Test states are an ndarray
        self.assertEqualC(states.shape)  # Test states have the right shape
        self.assertL2(states, tol=0.03)

    def test_direct_1(self):
        x_target = (2, 0, 0)
        self.chk_direct(x_target)

    def test_direct_2(self):
        x_target = (2, 2, np.pi / 2)
        self.chk_direct(x_target)


class R2D2Linearization(UTestCase):
    """  Problem 17: R2D2 and simple linearization """

    def chk_linearization(self, x_target):
        from irlc.project2.r2d2 import drive_to_linearization
        states = drive_to_linearization(x_target=x_target, plot=False)
        self.assertIsInstance(states, np.ndarray)  # Test states are an ndarray
        self.assertEqualC(states.shape)  # Test states have the right shape
        self.assertL2(states, tol=0.03)

    def test_linearization_1(self):
        x_target = (2, 0, 0)
        self.chk_linearization(x_target)

    def test_linearization_2(self):
        x_target = (2, 2, np.pi / 2)
        self.chk_linearization(x_target)

class R2D2_MPC(UTestCase):
    """ Problem 18: R2D2 and MPC """

    def chk_mpc(self, x_target):
        from irlc.project2.r2d2 import drive_to_mpc
        states = drive_to_mpc(x_target=x_target, plot=False)
        self.assertIsInstance(states, np.ndarray)  # Test states are an ndarray
        self.assertEqualC(states.shape)  # Test states have the right shape
        self.assertL2(states, tol=0.03)

    def test_mpc_1(self):
        self.chk_mpc(x_target=(2,0,0) )

    def test_mpc_2(self):
        self.chk_mpc(x_target=(2, 2, np.pi / 2))

class Project2(Report):
    title = "Project part 2: Control"
    pack_imports = [irlc]

    yoda1 = [
        (YodaProblem1, 10),
        (YodaProblem3, 10),
        (YodaProblem4, 10),
             ]
    yoda2 = [
        (YodaProblem6, 10),
        (YodaProblem7, 10),
        (YodaProblem11, 10),
        (YodaProblem12, 2)
             ]
    r2d2 = [
            (R2D2Problem15, 10),
            (R2D2Direct, 10),
            (R2D2Linearization, 10),
            (R2D2_MPC, 10),
            ]

    questions = []
    questions += yoda1
    questions += yoda2
    questions += r2d2

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Project2() )

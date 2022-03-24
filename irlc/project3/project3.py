# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, Report
import irlc

class JarJarPiOptimal(UTestCase):
    """ Problem 1: Compute optimal policy.  """
    def test_pi_1(self):
        from irlc.project3.jarjar import pi_optimal
        self.assertEqual(pi_optimal(1), -1)

    def test_pi_all(self):
        from irlc.project3.jarjar import pi_optimal
        for s in range(-10, 10):
            if s != 0:
                self.assertEqualC(pi_optimal(s))

class JarJarQ0Estimated(UTestCase):
    """ Problem 2: Implement Q0_approximate to (approximate) the Q-function for the optimal policy.  """
    def test_Q0_N1(self):
        from irlc.project3.jarjar import Q0_approximate
        self.assertEqualC(Q0_approximate(gamma=0.8, N=1))

    def test_Q0_N2(self):
        from irlc.project3.jarjar import Q0_approximate
        self.assertEqualC(Q0_approximate(gamma=0.7, N=20))

    def test_Q0_N100(self):
        from irlc.project3.jarjar import Q0_approximate
        self.assertEqualC(Q0_approximate(gamma=0.9, N=20))


class JarJarQExact(UTestCase):
    """ Problem 4: Compute Q^*(s,a) exactly by extending analytical solution. """
    def test_Q_s0(self):
        from irlc.project3.jarjar import Q_exact
        self.assertEqualC(Q_exact(0, gamma=0.8, a=1))
        self.assertEqualC(Q_exact(0, gamma=0.8, a=-1))

    def test_Q_s1(self):
        from irlc.project3.jarjar import Q_exact
        self.assertEqualC(Q_exact(1, gamma=0.8, a=-1))
        self.assertEqualC(Q_exact(1, gamma=0.95, a=-1))
        self.assertEqualC(Q_exact(1, gamma=0.7, a=-1))

    def test_Q_s_positive(self):
        from irlc.project3.jarjar import Q_exact
        for s in range(20):
            Q_exact(s, gamma=0.75, a=-1)

    def test_Q_all(self):
        from irlc.project3.jarjar import Q_exact
        for s in range(-20, 20):
            Q_exact(s, gamma=0.75, a=-1)
            Q_exact(s, gamma=0.75, a=1)



class RebelsSimple(UTestCase):
    """ Problem 5: Test the UCB-algorithm in the basic-environment with a single state """
    def test_simple_four_episodes(self):
        """ Test the first four episodes in the simple grid problem. """
        from irlc.project3.rebels import get_ucb_actions, very_basic_grid
        actions = get_ucb_actions(very_basic_grid, alpha=0.1, episodes=4, c=5, plot=False)
        # Make sure we only have 4 actions (remember to truncate the action-sequences!)
        self.assertEqual(len(actions), 4) # Check the number of actions are correct
        self.assertEqual(actions[0], 0) # Check the first action is correct
        self.assertEqualC(actions) # Check all actions.

    def test_simple_nine_episodes(self):
        """ Test the first nine episodes in the simple grid problem. """
        from irlc.project3.rebels import get_ucb_actions, very_basic_grid
        actions = get_ucb_actions(very_basic_grid, alpha=0.1, episodes=9, c=5, plot=False)
        self.assertEqual(len(actions), 9) # Check the number of actions are correct
        self.assertEqual(actions[0], 0) # Check the first action is correct
        self.assertEqualC(actions) # Check all actions.

    def test_simple_environment(self):
        from irlc.project3.rebels import get_ucb_actions, very_basic_grid
        actions = get_ucb_actions(very_basic_grid, alpha=0.1, episodes=100, c=5, plot=False)
        # Check the number of actions are correct
        self.assertEqualC(len(actions))
        # Check the first action is correct
        self.assertEqualC(actions[0])
        # Check all actions.
        self.assertEqualC(actions)

    def test_bridge_environment(self):
        from irlc.gridworld.gridworld_environments import grid_bridge_grid
        from irlc.project3.rebels import get_ucb_actions, very_basic_grid
        actions = get_ucb_actions(grid_bridge_grid, alpha=0.1, episodes=1000, c=2, plot=False)
        self.assertEqualC(len(actions))
        # Check all actions.
        self.assertEqualC(actions)

class RebelsBridge(UTestCase):
    """ Problem 5: Test the UCB-algorithm in the bridge-environment """
    def test_bridge_environment_one(self):
        from irlc.gridworld.gridworld_environments import grid_bridge_grid
        from irlc.project3.rebels import get_ucb_actions
        actions = get_ucb_actions(grid_bridge_grid, alpha=0.1, episodes=1, c=2, plot=False)
        self.assertEqualC(len(actions))
        self.assertEqualC(actions)

    def test_bridge_environment_two(self):
        from irlc.gridworld.gridworld_environments import grid_bridge_grid
        from irlc.project3.rebels import get_ucb_actions
        actions = get_ucb_actions(grid_bridge_grid, alpha=0.1, episodes=2, c=2, plot=False)
        self.assertEqualC(len(actions))
        self.assertEqualC(actions)

    def test_bridge_environment_short(self):
        from irlc.gridworld.gridworld_environments import grid_bridge_grid
        from irlc.project3.rebels import get_ucb_actions
        actions = get_ucb_actions(grid_bridge_grid, alpha=0.1, episodes=30, c=2, plot=False)
        self.assertEqualC(len(actions))
        self.assertEqualC(actions)

    def test_bridge_environment_long(self):
        from irlc.gridworld.gridworld_environments import grid_bridge_grid
        from irlc.project3.rebels import get_ucb_actions
        actions = get_ucb_actions(grid_bridge_grid, alpha=0.1, episodes=1000, c=2, plot=False)
        self.assertEqualC(len(actions))
        self.assertEqualC(actions)

class Project3(Report):
    title = "Project part 3: Reinforcement Learning"
    pack_imports = [irlc]

    jarjar1 = [(JarJarPiOptimal, 10),
               (JarJarQ0Estimated, 10),
               (JarJarQExact, 10) ]

    rebels = [(RebelsSimple, 20),
              (RebelsBridge, 20) ]
    questions = []
    questions += jarjar1
    questions += rebels

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Project3())

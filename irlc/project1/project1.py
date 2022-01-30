# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, Report
from irlc.pacman.gamestate import GameState
from irlc.pacman.pacman_environment import GymPacmanEnvironment
import numpy as np

def get_starting_state(name):
    from irlc.project1.pacman import east, east2, SS0tiny, SS1tiny, SS2tiny
    names2maps = {'east': east,
                  'east2': east2,
                  'SS0tiny': SS0tiny,
                  'SS1tiny': SS1tiny,
                  'SS2tiny': SS2tiny,
                  }

    return GymPacmanEnvironment(layout_str=names2maps[name]).reset()

def get_map(name):
    from irlc.project1.pacman import east, east2, SS0tiny, SS1tiny, SS2tiny
    names2maps = {'east': east,
                  'east2': east2,
                  'SS0tiny': SS0tiny,
                  'SS1tiny': SS1tiny,
                  'SS2tiny': SS2tiny,
                  }
    return names2maps[name]

class Pacman1(UTestCase):
    """ Problem 1: The go_east function """

    def test_states_length(self):
        from irlc.project1.pacman import go_east, east
        self.title = "Checking number of states"
        self.assertEqualC(len(go_east(east)))

    def test_first_state(self):
        from irlc.project1.pacman import go_east, east
        self.title = "Checking first state"
        self.assertEqualC(str(go_east(east))[0]) # string representation of the first state.

    def test_all_states(self):
        self.title = "Checking complete output"
        from irlc.project1.pacman import go_east, east
        self.assertEqualC(tuple(str(s) for s in go_east(east)))


class Pacman2(UTestCase):
    """ Problem 2: the p_next function """
    map = 'east'
    action = 'East'

    def get_transitions(self):
        from irlc.project1.pacman import p_next
        # state = GymPacmanEnvironment(layout_str=names2maps[self.map]).reset()
        state = get_starting_state(self.map)
        state_transitions = p_next(state, self.action)
        self.assertIsInstance(state_transitions, dict)
        for x in state_transitions:  # Test if each new state is actually a GameState.
            self.assertIsInstance(x, GameState)
        dd = {s: np.round(p, 4) for s, p in state_transitions.items()}
        return dd

    def test_dictionary_size(self):
        """ Is the number of keys/values in the dictionary correct? """
        self.assertEqualC(len(self.get_transitions()))

    def test_probabilities(self):
        """ Does the probabilities have the right value? """
        self.assertEqualC(tuple(self.get_transitions().values()))

    def test_states(self):
        """ Does the dictionary contains the right states """
        self.assertEqualC(tuple(self.get_transitions().keys()))

    def test_everything(self):
        """ Test both states and probabilities """
        self.assertEqualC(tuple(self.get_transitions()))


class Pacman3(UTestCase):
    """ Problem 3: Compute the state spaces as a list [S_0, ..., S_N] on the map 'east' using N = 7 """
    map = 'east'
    N = 7

    @property
    def states(self):
        return self.__class__.states_

    @classmethod
    def setUpClass(cls):
        from irlc.project1.pacman import get_future_states
        states = get_future_states(get_starting_state(cls.map), cls.N)
        assert isinstance(states, list)
        for S in states:
            assert isinstance(S, list)
            for s in S:
                assert isinstance(s, GameState)
        cls.states_ = [set(S) for S in states]

    def test_number_of_spaces(self):
        """ Check the list of state spaces has the right length. It should be N+1 long (S_0, ..., S_N) """
        self.assertEqualC(len(self.states))

    def test_state_space_0(self):
        """ Check the first element, the state space S0.

        Hints:
            * It should be a list containning a single GameState object (the starting state) """
        self.assertEqualC(self.states[0])

    def test_state_space_1(self):
        """ Check the second element, the state space S1.

        Hints:
            * It should be a list containing the GameState objects you can go to in one step.
            * You should be able to figure out what they are from the description of the game rules. Note pacman will not move if he walks into the walls. """
        self.assertEqualC(self.states[1])

    def test_state_spaces(self):
        """ Test all state spaces S_0, ..., S_N

        Hints:
            * If this method breaks, find the first state space which is wrongly computed, and work out which states are missing or should not be there
            * I anticipate the won/lost game configurations may become a source of problems. Note you don't have to specify these manually; they should follow by using the s.f(action)-function. """

        self.assertEqualC(tuple(self.states))


class Pacman4(UTestCase):
    """ Problem 4a: No ghost optimal path (get_shortest_path) in map 'east' using N=20 """
    map = 'east'
    N = 20

    def get_shortest_path(self):
        from irlc.project1.pacman import shortest_path
        layout = get_map(self.map)
        cost, actions, states = shortest_path(layout, self.N)
        return cost, actions, states

    def test_shortest_path_cost(self):
        """ Shortest path cost """
        cost, actions, states = self.get_shortest_path()
        self.assertEqualC(cost)

    def test_sequence_lengths(self):
        """ Test the length of the state/action lists. """
        cost, actions, states = self.get_shortest_path()
        self.assertEqualC(len(actions))
        self.assertEqualC(len(states))

    def test_trajectory(self):
        """ Test the state/action trajectory """
        J, actions, states = self.get_shortest_path()
        self.assertTrue(states[-1].isWin())

        x0 = states[0]
        for k, u in enumerate(actions):
            x0 = x0.f(u)
            self.assertTrue(x0 == states[k + 1])
        self.assertEqualC(states[1])
        self.assertEqualC(J)

class Pacman4b(Pacman4):
    """ Problem 4b: Determine the shortest path on the map SS0tiny """
    map = 'SS0tiny'

## ONE GHOST
class Pacman5(Pacman3):
    """ Problem 5:  This class test the state spaces as a list [S_0, ..., S_N]. on the map 'SS1tiny' using N = 4 """
    map = 'SS1tiny'
    N = 4

class Pacman6(Pacman3):
    """ Pacman6: This class test the state spaces as a list [S_0, ..., S_N]. on the map 'SS1tiny' using N = 6 """
    map = 'SS1tiny'
    N = 6
    pass

class Pacman7(UTestCase):
    """ Testing winrate on the map SS1tiny (win_probability) """
    map = 'SS1tiny'

    def _win_rate(self, N):
        self.title = "Testing winrate in N steps"
        from irlc.project1.pacman import win_probability
        p = np.round(win_probability(get_map(self.map), N), 4)
        self.assertEqualC(p)


    def test_win_rate_N4(self):
        self._win_rate(N=4)

    def test_win_rate_N8(self):
        self._win_rate(N=5)


# ## TWO GHOSTS
class Pacman8(Pacman2):
    map = 'SS2tiny'
    N = 4

class Pacman9(Pacman3):
    map = 'SS2tiny'
    N = 3

class Pacman10(Pacman7):
    map = 'SS2tiny'
    N = 2

class Kiosk1(UTestCase):
    """ Warmup check of S_0 and A_0(x_0) """
    def test_warmup_states_length(self):
        from irlc.project1.kiosk import warmup_states, warmup_actions
        n = len(warmup_states())
        self.title = f"Checking length of state space is {n}"
        self.assertEqualC(n)

    def test_warmup_actions_length(self):
        from irlc.project1.kiosk import warmup_states, warmup_actions
        n = len(warmup_actions())
        self.title = f"Checking length of action space is {n}"
        self.assertEqualC(n)


    def test_warmup_states(self):
        self.title = "Checking state space"
        from irlc.project1.kiosk import warmup_states, warmup_actions
        self.assertEqualC(set(warmup_states()))

    def test_warmup_actions(self):
        self.title = "Checking action space"
        from irlc.project1.kiosk import warmup_states, warmup_actions
        self.assertEqualC(set(warmup_actions()))


class Kiosk2(UTestCase):
    """ solve_kiosk_1 """

    @classmethod
    def setUpClass(cls) -> None:
        from irlc.project1.kiosk import solve_kiosk_1
        cls.J, cls.pi = solve_kiosk_1()

    def mk_title(self, k, x):
        self.k = k
        self.x = x

        if self.k is not None:
            if self.k != -1:
                sk = f"N-{-self.k - 1}" if self.k < 0 else str(self.k)
            else:
                sk = "N"
            jp = "J_{" + sk + "}" if len(sk) > 1 else "J_"+sk
        else:
            jp = "J_k"
        if self.x is not None:
            xp = f"(x={self.x})"
        else:
            xp = "(x) for all x"
        return "Checking cost-to-go " + jp + xp
        pass

    def check_J(self, k, x):
        J = [{k: np.round(v, 4) for k, v in J_.items()} for J_ in Kiosk2.J]
        t = self.mk_title(k, x)
        if k is not None and x is not None:
            t += f" = {J[k][x]}"
        self.title = t

        if k is not None:
            J_ = J[k]
            if x is not None:
                value = J_[x]
            else:
                value = J_
        else:
            value = J

        self.assertEqualC(value)

    def test_case_1(self):
        self.check_J(k=-1, x=10)

    def test_case_2(self):
        self.check_J(k=-2, x=20)

    def test_case_3(self):
        self.check_J(k=-2, x=0)

    def test_case_4(self):
        self.check_J(k=0, x=0)

    def test_case_5(self):
        self.check_J(k=None, x=None)


class Kiosk3(Kiosk2):
    """ solve_kiosk_2 """
    @classmethod
    def setUpClass(cls) -> None:
        from irlc.project1.kiosk import solve_kiosk_2
        cls.J, cls.pi = solve_kiosk_2()


class Kiosk4(Kiosk2):
    """ solve_kiosk_3 """

    @classmethod
    def setUpClass(cls) -> None:
        from irlc.project1.kiosk import solve_kiosk_3
        cls.J, cls.pi = solve_kiosk_3()


class Kiosk5(Kiosk2):
    title = "solve_kiosk_4"
    def get_kiosk_Jpi(self):
        from irlc.project1.kiosk import solve_kiosk_4
        return solve_kiosk_4()

class Project1(Report): #240 total.
    title = "02465 project part 1: Dynamical Programming"
    import irlc
    pack_imports = [irlc]
    abbreviate_questions = True


    pacman_questions = [
        (Pacman1, 10),
        (Pacman2, 10),
        (Pacman3, 10),
        (Pacman4, 10),
        (Pacman4b, 10),
        (Pacman5, 10),
        (Pacman6, 10),
        (Pacman7, 10),
        (Pacman8, 10),
        (Pacman9, 10),
        (Pacman10, 10),
                 ]

    kiosk_questions = [
        (Kiosk1, 10),
        (Kiosk2, 10),
        (Kiosk3, 10),
        (Kiosk4, 10),
        (Kiosk5, 10),
    ]

    questions = []
    questions += pacman_questions
    questions += kiosk_questions

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Project1())
# 448, 409 # 303,

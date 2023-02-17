# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her23] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2023.
"""
from irlc.ex02.graph_traversal import G222


class SearchProblem: 
    """
    This class represents a search problem as defined in (Her23, Theorem 7.2.1).

    The search problem has a function to determine if a ``state``, and a function that computes the available transitions.

    To use this class, you should overwrite these two functions, similar to how you used the ``DPModel`` class.
    An example that does this can be found in the :class:`irlc.ex03.search_problem.GraphSP` class found below. To use this class you can do

    >>> sp = GraphSP()
    >>> sp.is_terminal(3) # In this case the states are integers, 0, 1,.., corresponding to nodes in the graph.
    >>> print(sp.available_transitions(2)) # Dictionary of the form {action: states, ...}

    """
    def __init__(self, initial_state=None):
        """
        Initialize the search problem. By default, the search problem is given an initial state, as very often we want
        to be able to change the initial state for the same problem.

        :param initial_state: The initial (starting) state of the search problem.
        """
        if initial_state is not None:
            self.set_initial_state(initial_state)

    def set_initial_state(self, state):
        """
        Re-set the initial (starting) state of the search problem.
        This is a convenience function you don't have to implement/overwrite.

        :param state: The new initial state of the search problem.
        :return: nothing
        """
        self.initial_state = state  # Re-set the initial state

    def is_terminal(self, state):
        """
        Determines if the given ``state`` is terminal.

        :param state: A valid state for the search problem
        :return:  ``True`` if ``state`` is a terminal state and otherwise ``False``
        """
        raise NotImplementedError("Implement a goal test")

    def available_transitions(self, state):
        """
        Returns the available transitions in this state as a dictionary ``{a: (s1, c), a2: (s2,c), ...}``

          - The keys in the dictionary are actions that can be taken in ``state``
          - The values are tuples ``(next_state, cost)``

            - ``next_state`` is the state we transition to upon taking the given action
            - ``cost`` is the specific cost incurred by that transition

        In graph search problem ``GraphSP``, the ``actions`` are equal to the next state since there is only a single way to transition from
        one state to another.

        .. tip::
            The following shows an example using the ``GraphSP`` class.

            .. doctest::

                >>> model = GraphSP(start=2, goal=5)
                >>> model.is_terminal(1) # Is the state `1` terminal? (No: Only state ``5`` is terminal as this is the goal state)
                True
                >>> transitions = model.available_transitions(2)
                >>> sp, cost = transitions[1] # next state and cost if action 1 is taken (note sp=1).

        .. Note::
            This formulation is required since in e.g. Pacman, multiple actions can lead to the same (next) state.
            For instance, if we try to walk into a wall, we will be blocked similar to what happened if we take the stop-action.


        :param state: The state we want to determine the transitions in
        :return: A dictionary where keys are available actions and values are the next state and cost obtained by taking that action.
        """
        raise NotImplementedError("Transition function not impelmented") 

class EnsureTerminalSelfTransitionsWrapper(SearchProblem):
    def __init__(self, search_problem):
        self._sp = search_problem
        super().__init__(search_problem.__dict__.get('initial_state', None)) # Get initial state if set.

    def set_initial_state(self, state):
        self._sp.set_initial_state(state)

    @property
    def initial_state(self):
        return self._sp.initial_state

    def is_terminal(self, state):
        return self._sp.is_terminal(state)

    def available_transitions(self, state):
        return {0: (state, 0)} if self.is_terminal(state) else self._sp.available_transitions(state)

class DP2SP(SearchProblem):
    """ This class converts a Deterministic DP environment to a shortest path problem matching the description
    in (Her23, eq. (5.16)).
    """
    def __init__(self, env, initial_state):
        self.env = env
        self.terminal_state = "terminal_state"
        super(DP2SP, self).__init__(initial_state=(initial_state, 0))

    def is_terminal(self, state):
        return state == self.terminal_state

    def available_transitions(self, state):
        """ Implement the dp-to-search-problem conversion described in (Her23, Theorem 7.2.2). Keep in mind the time index is
        absorbed into the state; this means that state = (x, k) where x and k are intended to be used as
        env.f(x, <action>, <noise w>, k).
        As usual, you can set w=None since the problem is assumed to be deterministic.

        The output format should match a SearchProblem, i.e. a dictionary with keys as u and values as (next_state, cost):
            return {u1: (s_next_1, cost_1),
                    u2: (s_next_2, cost_2),
                    ... }
        """
        if state == self.terminal_state:
            return {0: (self.terminal_state, 0)}
        s, k = state
        if k == self.env.N:
            return {0: ( self.terminal_state, self.env.gN(s))}
        # TODO: 1 lines missing.
        raise NotImplementedError("return transtitions as a dictionary. Note the 1-line solution requires dictionary comprehension.")

class SmallGraphSP(SearchProblem):
    G = G222


class GraphSP(SearchProblem): 
    """ Implement the small graph graph problem in (Her23, Subsection 5.1.1) """
    G = G222

    def __init__(self, start=2, goal=5):
        self.goal = goal
        super().__init__(initial_state=start)

    def is_terminal(self, state): # Return true if the state is a terminal state
        return state == self.goal

    def available_transitions(self, i):
        # In vertex i, return available transitions i -> j and their cost.
        # This is encoded as a dictionary, such that the keys are the actions, and
        # the values are of the form (next_state, cost).
        return {j: (j, cost) for (i_, j), cost in self.G.items() if i_ == i} 

    @property
    def vertices(self):
        # Helper function: Return number of vertices in the graph. You can ignore this.
        return len(set([i for edge in self.G for i in edge]))

# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her23] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2023.
"""
import numpy as np
from irlc.ex02.dp_model import DPModel
from irlc.ex02.graph_traversal import symG
from irlc.ex03.search_problem import DP2SP
from irlc.ex03.dp_forward import dp_forward

Gtravelman = {("A", "B"): 5, ("A", "C"): 1, ("A", "D"): 15, ("B", "C"): 20, ("B", "D"): 4, ("C", "D"): 3}
symG(Gtravelman)  # make graph symmetric.

class TravelingSalesman(DPModel):
    """ Travelling salesman problem, see (Her23, Subsection 7.1.1)
    Visit all nodes in the graph with the smallest cost, and such that we end up where we started.
    The actions are still new nodes we can visit, however the states have to be the current path.

    I.e. the first state is s = ("A", ) and then the next state could be s = ("A", "B"), and so on.
    """
    def __init__(self):
        self.G = Gtravelman
        self.cities = {c for chord in self.G for c in chord}
        N = len(self.cities)
        super(TravelingSalesman, self).__init__(N)

    def f(self, x, u, w, k):
        assert((x[-1],u) in self.G)
        ## TODO: Half of each line of code in the following 1 lines have been replaced by garbage. Make it work and remove the error.
        #----------------------------------------------------------------------------------------------------------------------------
        # return ????????
        raise NotImplementedError("Insert your solution and remove this error.")

    def g(self, x, u, w, k):
        ## TODO: Half of each line of code in the following 1 lines have been replaced by garbage. Make it work and remove the error.
        #----------------------------------------------------------------------------------------------------------------------------
        # return self????????????
        raise NotImplementedError("Insert your solution and remove this error.")

    def gN(self, x):
        """
        Win condition is that:

        (1) We end up where we started AND
        (2) we saw all cities AND
        (3) all cities connected by path

        If these are met return 0 otherwise np.inf
        """
        ## TODO: Half of each line of code in the following 1 lines have been replaced by garbage. Make it work and remove the error.
        #----------------------------------------------------------------------------------------------------------------------------
        # path_ok = x[0] == x[-1] and len(set(x)) == len(self.cities)????????????????????????????????????????????????????????????
        raise NotImplementedError("Insert your solution and remove this error.")
        return 0 if path_ok else np.inf

    def A(self, x, k):
        return {b for (a, b) in self.G if x[-1] == a}

def main():
    tm = TravelingSalesman()
    s = ("A",)
    tm_sp = DP2SP(tm, s)
    J, actions, path = dp_forward(tm_sp, N=tm.N)
    print("Cost of optimal path (should be 13):", J[-1][tm_sp.terminal_state])
    print("Optimal path:", path)
    print("(Should agree with (Her23, Subsection 7.1.1))")  

if __name__ == "__main__":
    main()

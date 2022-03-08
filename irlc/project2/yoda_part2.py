# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from scipy.linalg import expm  # Computes the matrix exponential e^A for a square matrix A
from numpy.linalg import matrix_power  # Computes A^n for matrix A and integer n
from irlc.project2.yoda_part1 import get_A_B


def A_euler(g,L,Delta): 
    """ Compute \tilde{A}_0 (Euler discretization), see Problem 6.

    Hints:
        * get_A_B can perhaps save you a line or two.
    """
    # TODO: 2 lines missing.
    raise NotImplementedError("Implement function body")
    return A0_tilde

def A_ei(g,L,Delta): 
    """ Compute A_0 (Exponential discretization), see Problem 6

    Hints:
        * The special function expm(X) computes the matrix exponential e^X. See lecture notes for more information.
    """
    # TODO: 2 lines missing.
    raise NotImplementedError("Implement function body")
    return A0

def M_euler(g, L, Delta, N): 
    """ Compute \tilde{M} (Euler discretization), see Problem 7
    Hints:
        * the matrix_power(X,n) function can compute expressions such as X^n where X is a square matrix and n is a number
    """
    # TODO: 1 lines missing.
    raise NotImplementedError("Implement function body")
    return M_tilde

def M_ei(g,L,Delta,N): 
    """ Compute M (Exponential discretization), see Problem 7 """
    # TODO: 1 lines missing.
    raise NotImplementedError("Implement function body")
    return M

def xN_bound_euler(g,L,Delta,N): 
    """ Compute upper bound on |x_N| when using Euler discretization, see Problem 10.
    The function should just return a number.

    Hints:
        * This function uses all input arguments.
    """
    # TODO: 1 lines missing.
    raise NotImplementedError("Implement function body")
    return bound

def xN_bound_ei(g,L,Delta,N): 
    """ Compute upper bound on |x_N| when using exponential discretization, see Problem 11.

    Hints:
        * This function does NOT use all input arguments.
        * This will be the hardest problem to solve, but the easiest function to implement.
    """
    # TODO: 1 lines missing.
    raise NotImplementedError("Implement function body")
    return bound

if __name__ == '__main__':
    g = 9.82 # graviational constant
    L = 5 # Length of string
    Delta = 0.3 # Time-discretization constant (seconds)
    m = 0.1 # Mass of pendulum
    N = 100 # Time steps

    # Solve Problem 6
    print("A0_euler")
    print(A_euler(g, L, Delta))

    print("A0_ei")
    print(A_ei(g, L, Delta))

    # Solve Problem 7
    print("M_euler")
    print(M_euler(g,L,Delta,N))

    print("M_ei")
    print(M_ei(g, L, Delta, N))

    # Solve Problem 11, upper bound on x_N using Euler discretization
    print("|x_N| <= ", xN_bound_euler(g, L, Delta, N))

    # Solve Problem 12, upper bound on x_N using Exponential discretization
    print("|x_N| <= ", xN_bound_ei(g, L, Delta, N))

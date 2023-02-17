# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
This project resembles the Inventory-control problem discussed in (Her23, Subsection 5.1.2) but with more complicated rules.
If you are stuck, the inventory-control problem will be a good place to start.

I recommend to use the DP_stochastic function (as we did with the inventory-control example). This means
your main problem is to build appropriate DPModel-classes to represent the different problems.
References:
  [Her23] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2023.
"""
from irlc.ex02.dp_model import DPModel
from irlc.ex02.dp import DP_stochastic
import matplotlib.pyplot as plt
from scipy.stats import binom
from irlc import savepdf
import numpy as np

def plot_policy(pi, title, pdf):
    """ Helper function to plot the policy functions pi, as generated by the DP_stochastic function. This function
    can be used to visualize which actions are taken in which state (y-axis) at which time step (x-axis). """
    N = len(pi)
    W = max(pi[0].keys())
    A = np.zeros((W, N))
    for i in range(W):
        for j in range(N):
            A[i, j] = pi[j][i]
    plt.imshow(A)
    plt.title(title)
    savepdf(pdf)
    plt.show()

# TODO: 51 lines missing.
raise NotImplementedError("Insert your solution and remove this error.")

def warmup_states(): 
    # TODO: 1 lines missing.
    raise NotImplementedError("return state set")

def warmup_actions(): 
    # TODO: 1 lines missing.
    raise NotImplementedError("return action set")

def solve_kiosk_1(): 
    # TODO: 1 lines missing.
    raise NotImplementedError("Return cost and policy here (same format as DP_stochastic)")

def solve_kiosk_2(): 
    # TODO: 1 lines missing.
    raise NotImplementedError("Return cost and policy here (same format as DP_stochastic)")


def main():
    # Problem 14
    print("Available states S_0:", warmup_states())
    print("Available actions A_0(x_0):", warmup_actions())

    J, pi = solve_kiosk_1() # Problem 16
    print("Kiosk1: Expected profits: ", -J[0][0], " imperial credits")
    plot_policy(pi, "Kiosk1", "Latex/figures/kiosk1")
    plt.show()

    J, pi = solve_kiosk_2() # Problem 17
    print("Kiosk 2: Expected profits: ", -J[0][0], " imperial credits")
    plot_policy(pi, "Kiosk2", "Latex/figures/kiosk2")
    plt.show()


if __name__ == "__main__":
    main()

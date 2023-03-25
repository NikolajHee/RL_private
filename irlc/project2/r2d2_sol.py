# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
# import sympy as sym
# from scipy.optimize import Bounds
# from gym.spaces import Box
# import matplotlib.pyplot as plt
# from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
# from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment
# from irlc.ex04.continuous_time_model import ContiniousTimeSymbolicModel
# from irlc.ex04.cost_continuous import SymbolicQRCost
# from irlc.ex05.direct_agent import DirectAgent
# from irlc.ex05.direct import get_opts
# from irlc.ex06.linearization_agent import LinearizationAgent
# from irlc.project2.utils import render_car
# from irlc import VideoMonitor, Agent, train, plot_trajectory, savepdf


if __name__ == '__main__':
    from irlc.project2.r2d2 import linearize
    theta = 0.911
    xbar = np.asarray([0, 0, theta])
    ubar_init = np.asarray([1, 0])
    Delta = 0.1532
    A, B, d = linearize(xbar, ubar_init, Delta=Delta)

    dA = np.asarray([[1, 0, -Delta * np.sin(theta)],
                     [0, 1, Delta * np.cos(theta)],
                     [0, 0, 1.]])

    dB = np.asarray([[Delta * np.cos(theta), 0],
                     [Delta * np.sin(theta), 0],
                     [0, Delta]])

    dd = np.asarray([Delta * np.sin(theta)*theta,
                     -Delta * np.cos(theta)*theta,
                     0])

    np.allclose(B, dB)
    np.allclose(A, dA)
    np.allclose(d, dd)

    a = 23
    pass

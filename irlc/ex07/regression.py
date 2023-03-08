# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her23] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2023.
"""
import numpy as np
import scipy.linalg as linalg


def solve_linear_problem_simple(Y, X, U, lamb=0): 
    """
    This function uses linear regression to find matrices and vectors :math:`A`, :math:`B` and :math:`d` in the linear dynamics:

    .. math::
      x_{k+1} = A x_k + B u_k + d 

    This is accomplished by gathering (stacking) all observations vertically as follows:

        - all :math:`N` observations of :math:`x_{k+1}` into an :math:`N \\times n` matrix :math:`Y`,
        - all :math:`N` observations of :math:`x_{k}` into an :math:`N \\times n` matrix :math:`X`,
        - all :math:`N` observations of :math:`u_{k}` into an :math:`N \\times d` matrix :math:`U`,

    which allows us to write the problem in a more condensed format:

    .. math::
      Y^\\top = AX^\\top + BU^\\top + d

    which can be solved using multi-dimensional linear regression. The method specifically implements \(Her23, Algorithm 26).

    :param Y: A :math:`N \\times n` numpy ndarray
    :param X: A :math:`N \\times n` numpy ndarray
    :param U: A :math:`N \\times d` numpy ndarray
    :param lamb: Regularization strength :math:`\lambda`
    :return:
        - A - A :math:`n\\times n` numpy ``ndarray``
        - B - A :math:`n\\times d` numpy ``ndarray``
        - d - A :math:`n\\times 1` numpy ``ndarray``
    """ 
    Y = Y.T
    X = X.T
    U = U.T
    n,d = X.shape[0], U.shape[0]
    P_list = [np.eye(n), np.eye(n), np.eye(n)]
    Z_list = [X, U, np.ones( (1,X.shape[1]))]
    W = solve_linear_problem(Y, Z_list=Z_list, P_list=P_list, lamb=lamb)
    A, B, d = W[0], W[1], vec(W[2])
    return A, B, d 

def vec(M):
    return M.flatten('F')

def solve_linear_problem(Y, Z_list, P_list=None, lamb=0, weights=None):
    if P_list is None:
        P_list = [np.eye(Z_list[0].shape[0])]

    if weights is None:
        weights = np.ones( (Y.shape[1],) )

    Sigma = linalg.kron( np.diag(weights), np.eye( Y.shape[0] ) )
    S = np.concatenate( [linalg.kron(Z.T, P) for Z,P in zip(Z_list, P_list) ], axis=1)
    Wvec = np.linalg.solve( (S.T @ Sigma @ S + lamb * np.eye(S.shape[1])), S.T @ Sigma.T @ vec(Y))
    # unstack
    W = []
    d0 = 0
    for Z,P in zip(Z_list, P_list):
        dims = (P.shape[1], Z.shape[0])
        Wj = np.reshape( Wvec[d0:d0+np.prod(dims)], newshape=dims, order='F')
        d0 += np.prod(dims)
        W.append(Wj)
    return W

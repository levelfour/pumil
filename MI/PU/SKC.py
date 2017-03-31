#!/usr/bin/env python
# coding: utf-8

import MI
import numpy as np


_SOLVER = 'cvxopt'


def train(bags, basis, bdim, theta, r, args):
  if args.loss == 'double_hinge':
    return train_dh(bags, basis, bdim, theta, r, args)

  elif args.loss == 'squared':
    return train_sl(bags, basis, bdim, theta, r, args)


def train_sl(bags, basis, bdim, theta, r, args):
  p_bags = MI.extract_bags(bags, 1)
  u_bags = MI.extract_bags(bags, 0)
  N1 = len(p_bags)
  N0 = len(u_bags)
  N = N1 + N0
  P1 = np.array([np.r_[1, basis(B)].T for B in p_bags])
  P0 = np.array([np.r_[1, basis(B)].T for B in u_bags])

  param = np.linalg.inv(0.5 / N0 * P0.T.dot(P0) + r * np.eye(bdim + 1)).dot( \
      theta / N1 * P1.T.dot(np.ones((N1, 1))) - 0.5 / N0 * P0.T.dot(np.ones((N0, 1)))
  )

  alpha = param[1:]
  beta  = float(param[:1])
  clf   = lambda X: alpha.T.dot(basis(X)) + beta

  return clf


def train_dh(bags, basis, bdim, theta, r, args):
  if _SOLVER == 'cvxopt':
    import cvxopt
    from cvxopt import matrix
    from cvxopt.solvers import qp
    cvxopt.solvers.options['show_progress'] = False

  elif _SOLVER == 'openopt':
    from openopt import QP
    import warnings
    warnings.simplefilter(action = "ignore", category = FutureWarning)

  p_bags = MI.extract_bags(bags, 1)
  u_bags = MI.extract_bags(bags, 0)
  N1 = len(p_bags)
  N0 = len(u_bags)
  N = N1 + N0
  d = bdim
  P1 = np.array([basis(B).T for B in p_bags])
  P0 = np.array([basis(B).T for B in u_bags])
  H = np.r_[
      np.c_[r*np.eye(d),       np.zeros((d, 1)),  np.zeros((d, N0))],
      np.c_[np.zeros((1, d)),  0,                 np.zeros((1, N0))],
      np.c_[np.zeros((N0, d)), np.zeros((N0, 1)), np.zeros((N0, N0))]]
  f = np.r_[
      -theta/N1*P1.T.sum(axis=1).reshape((-1, 1)),
      [[-theta]],
      1./N0*np.ones((N0, 1))]
  L = np.r_[
      np.c_[           0.5*P0, 0.5*np.ones((N0, 1)), -np.eye(N0)],
      np.c_[               P0,     np.ones((N0, 1)), -np.eye(N0)],
      np.c_[np.zeros((N0, d)),    np.zeros((N0, 1)), -np.eye(N0)]]
  k = np.r_[
      -0.5*np.ones((N0, 1)),
      np.zeros((N0, 1)),
      -np.zeros((N0, 1))]

  if _SOLVER == 'cvxopt':
    result = qp(matrix(H), matrix(f), matrix(L), matrix(k))
    gamma = np.array(result['x'])

  elif _SOLVER == 'openopt':
    problem = QP(H + 1e-3 * np.eye(H.shape[0]), f, A = L, b = k)
    result  = problem.solve('qlcp')
    gamma = result.xf

  alpha = gamma[:d]
  beta  = gamma[d]
  clf   = lambda X: alpha.T.dot(basis(X)) + beta

  return clf

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import itertools
import MI


def nsk_basis(bags, width = 1.0e-01):
  """
  Build basis function based on normalized set kernel.
  """

  ins_kern = lambda x, c: np.exp(- width * np.linalg.norm(x - c) ** 2)

  p_bags = MI.extract_bags(bags, 1)
  u_bags = MI.extract_bags(bags, 0)
  n_bags = MI.extract_bags(bags, -1)
  bags = p_bags + u_bags + n_bags

  # (un-normalized) set kernel
  usk = lambda S0, S1: sum(list(map(
    lambda s: ins_kern(s[0], s[1]),
    list(itertools.product(S0, S1)))))

  # normalized set kernel
  nsk = lambda S0, S1: usk(S0, S1) / np.sqrt(usk(S0, S0) * usk(S1, S1))

  return lambda X: np.array([nsk(X, B) for B in bags])


def minimax_basis(bags, degree = 1):
  """
  Build basis function based on minimax kernel.

  Parameters
  ----------
  deg : Degree of polynomial kernel.
  """
  degree = int(degree)

  p_bags = MI.extract_bags(bags, 1)
  u_bags = MI.extract_bags(bags, 0)
  n_bags = MI.extract_bags(bags, -1)
  bags = p_bags + u_bags + n_bags

  stat = lambda X: np.r_[X.min(axis=0), X.max(axis=0)]
  poly_kern = lambda X, Y: (stat(X).dot(stat(Y)) + 1) ** degree

  return lambda X: np.array([poly_kern(X, B) for B in bags])

#!/usr/bin/env python
# coding: utf-8

import sys
import random
import numpy as np
import MI

def prepare(bags, class_prior, L, U, T):
  """
  Parameters
  ----------
  bags        : original dataset
  class_prior : the ratio of positive samples
  L           : the number of labeled samples in output dataset
  U           : the number of unlabeled samples in output dataset
  T           : the number of test samples in output dataset
  """
  # original data
  p_bags = MI.extract_bags(bags,  1, with_label = True)
  n_bags = MI.extract_bags(bags, -1, with_label = True)
  random.shuffle(p_bags)
  random.shuffle(n_bags)
  P = len(p_bags)
  N = len(n_bags)

  retry_count = 0
  while retry_count < 5:
    try:
      return _prepare(p_bags, n_bags, P, N, class_prior, L, U, T)
    except:
      # if the obtained split is invalid, try sampling again
      sys.stderr.write("Warning: Retry train-test-split (recommend to change the splitting number)\n")
      retry_count += 1
      continue

def _prepare(p_bags, n_bags, P, N, class_prior, L, U, T):
  # choose L positive samples
  # ( p_bags -> [  L  |   P - L (used for unlabeled set)  ] )
  assert L < P
  labeled_set = p_bags[:L]

  # make unlabeled set
  # [ (prior * U) from p_bags ]  + [((1 - prior) * U) from n_bags]
  UP = np.random.binomial(U + T, class_prior)
  assert UP < (P - L), "Insufficient positive samples"
  assert (U+T-UP) < N, "Insufficient negative samples"
  unlabeled_set = p_bags[L:(L+UP)] + n_bags[:(U+T-UP)]
  random.shuffle(unlabeled_set)

  # train test split
  train_unlabeled_set = unlabeled_set[:U]
  test_unlabeled_set  = unlabeled_set[U:]

  for bag in train_unlabeled_set:
    bag.mask()

  metadata = {
    'train_lp': L,
    'train_up': len(list(filter(lambda B: B.y == +1, train_unlabeled_set))),
    'train_un': len(list(filter(lambda B: B.y == -1, train_unlabeled_set))),
    'test_p'  : len(list(filter(lambda B: B.y == +1, test_unlabeled_set))),
    'test_n'  : len(list(filter(lambda B: B.y == -1, test_unlabeled_set))),
  }

  # (training_bags, test_bags)
  train = labeled_set + train_unlabeled_set
  test = test_unlabeled_set
  random.shuffle(train)
  random.shuffle(test)
  return train, test, metadata


def class_prior(data, degree, reg):
  basis = MI.kernel.minimax_basis(data, degree)
  return _class_prior(data, basis, reg)


def _class_prior(bags, basis, r):
  # cf. (du Plessis et al., 2014)
  p_bags = MI.extract_bags(bags, 1)
  u_bags = MI.extract_bags(bags, 0)
  n1 = len(p_bags)
  n0 = len(u_bags)
  H = 1./n1 * np.sum([np.outer(basis(B), basis(B).T) for B in p_bags], axis=0)
  h = 1./n0 * np.sum(list(map(lambda B: basis(B), u_bags)), axis=0)
  G = H + r * np.eye(n1 + n0)
  G_ = np.linalg.inv(G)
  return (2*h.T.dot(G_.dot(h))-h.T.dot(G_.dot(H.dot(G_.dot(h)))))**(-1)


from MI.PU import SKC
from MI.PU.loss import prediction_error

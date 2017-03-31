#!/usr/bin/env python
# coding: utf-8

import random
import itertools
import argparse
import MI
import numpy as np
from scipy import stats
from sklearn import svm

pca_dim = 30


def reliable_negative_bag_idx(bags, uidx, w, N):
  sorted_confs = sorted(list(zip(uidx, w[uidx])), key=lambda x:x[1], reverse=True)
  return [i for i, _ in sorted_confs[:N]]


def weighted_kde(Bn, conf):
  # N.B. apply WKDE function to the instance multiplied by its confindence value
  weighted_ins = np.vstack([conf[i] * B.data() for i, B in enumerate(Bn)])
  return stats.gaussian_kde(weighted_ins.T)


def form_pmp(bags, conf, pidx, nidx, Dn):
  choose_witness = lambda bags, conf: np.array([
    # for each bag
    bags[i].data()[
      # choose the least negative instance
      min(
        [(j, Dn(conf[i] * x)) for j, x in enumerate(bags[i].data())],
        key = lambda pair: pair[1]
      )[0]
    ]
    for i in range(len(bags))
  ])

  p_bags = [bags[i] for i in pidx]
  n_bags = [bags[i] for i in nidx]
  p_conf = [conf[i] for i in pidx]
  n_conf = [conf[i] for i in nidx]

  X = np.r_[choose_witness(p_bags, p_conf), choose_witness(n_bags, n_conf)]
  Y = np.r_[np.ones(len(p_bags)), -1 * np.ones(len(n_bags))]
  W = np.r_[p_conf, n_conf]

  return X, Y, W


def pumil_clf_wrapper(clf, n_dist, learning_phase = False):
  """
  Parameters
  ----------
  clf : instance classifier
  n_dist : (estimated) distribution of negative instances
  """
  witness = lambda xs, w: xs[
    min(
      [(j, n_dist(w * x)) for j, x in enumerate(xs)],
      key = lambda pair: pair[1]
    )[0]
  ].reshape(1, -1)

  if learning_phase:
    return lambda bag, conf: clf(witness(bag.data(), conf))

  else:
    # N.B. fix the confidence of test bag to 1
    return lambda instances: clf(witness(instances, 1))


def affinity(clf, conf, bags, uidx, nidx):
  # evaluate F-score on unlabeled set
  # regard "reliable negative bags" as negative set, and the other bags as positive set
  pidx = list(set(uidx) - set(nidx))
  pred = np.array([clf(bags[i], conf[i]) for i in pidx + nidx])
  true = np.r_[np.ones(len(pidx)), -1 * np.ones(len(nidx))]

  return MI.f_score(pred, true)


def train_pumil_clf(bags, pidx, uidx, w, NL, learning_phase = False):
  # top-{NL} reliable negative bags
  relnidx = reliable_negative_bag_idx(bags, uidx, w, NL)
  Bn = [bags[j] for j in relnidx]
  # estimated p(X|Y=-1) via WKDE
  Dn = weighted_kde(Bn, w[relnidx])
  # form Positive Margin Pool (PMP)
  pmp_x, pmp_y, pmp_conf = form_pmp(bags, w, pidx, relnidx, Dn)
  # train SVM by using PMP instances
  pmp_weighted_x = np.multiply(pmp_x.T, pmp_conf).T
  clf = svm.LinearSVC(loss = 'hinge')
  clf.fit(pmp_weighted_x, pmp_y)
  clf_ = pumil_clf_wrapper(lambda x: float(clf.decision_function(x)), Dn, learning_phase)

  if learning_phase:
    return clf_, relnidx

  else:
    return clf_


def pumil(bags, NL, NU, args):
  L = 10       # the number of confidence vectors
  c = 0.1      # clone factor
  T = 1.0e-03  # threshold (eps)
  M = 100      # max iteration limit

  # list of indices
  pidx = [i for i, B in enumerate(bags) if B.label() == 1]
  uidx = [i for i, B in enumerate(bags) if B.label() == 0]

  # initialization
  W = np.ones((L, NL + NU))
  for i in range(L):
    for j in uidx:
      # assign random confidences to unlabeled bags
      W[i][j] = np.random.uniform(0, 1)

  # training phase
  f = np.zeros(L)   # affinity scores for each confidence vector
  t = 1             # current epoch
  delta = 0         # the difference between best affinities of last and current epoch
  best_score = 0    # the best affinity score of the last epoch
  best_conf = None  # the best confidence weight vector

  while (t == 1) or (t <= M and delta >= T):
    for i in range(L):
      # obtain classifier from confidence vector
      clf, relnidx = train_pumil_clf(bags, pidx, uidx, W[i], NL, learning_phase = True)
      # calculate affinity scores
      f[i] = affinity(clf, W[i], bags, uidx, relnidx)

    best = f.argmax()

    # antibody clone
    for i in range(L):
      if i != best and stats.bernoulli(c) == 1:
        W[i] = W[best]

    # antibody mutation
    V = W + np.multiply(
      (W - np.tile(W[best], (L, 1))).T,
      (np.ones(L) - f) * np.random.uniform(0, 1, size=L)
    ).T

    # antibody update (whether to accept mutation or not)
    for i in range(L):
      # evaluate proposed confidence vector
      clf, relnidx = train_pumil_clf(bags, pidx, uidx, V[i], NL, learning_phase = True)
      f_ = affinity(clf, V[i], bags, uidx, relnidx)
      if f_ > f[i]:
        W[i] = V[i]
        f[i] = f_

    delta = f[best] - best_score
    best_score = f[best]
    best_conf  = W[best]

    t += 1

  return train_pumil_clf(bags, pidx, uidx, best_conf, NL)


DATASETS = [
    'synth',
    'musk1',
    'musk2',
    'fox',
    'elephant',
    'tiger',
    ]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="puMIL experiment toolkit")

  parser.add_argument('--dataset',
      action   = 'store',
      required = True,
      type     = str,
      choices  = DATASETS,
      help     = 'multiple instance dataset')

  parser.add_argument('--prior',
      action   = 'store',
      default  = 0.4,
      type     = float,
      metavar  = '[0-1]',
      help     = 'class prior (the ratio of positive data)')

  parser.add_argument('-v', '--verbose',
      action   = 'store_true',
      default  = False,
      help     = 'verbose output')

  parser.add_argument('--aucplot',
      action   = 'store_true',
      default  = False,
      help     = 'output prediction score and true label for AUC plot')

  args = parser.parse_args()

  if args.verbose:
    print("""# {}
# *** Experimental Setting ***
#   model                     : puMIL""".format('-'*80))
    print("#   dataset                   : {} (class prior = {})".format(args.dataset, args.prior))
    print("# !!NOTICE!! To keep WKDE working well, all instances are reduced to {}-dims by PCA".format(pca_dim))
    print("# {}".format('-'*80))

  # load data (dimension reduced to [pca_dim] for WKDE to work well)
  bags_train, bags_test, metadata = MI.datasets.load_dataset(args.dataset, args.prior, dim = pca_dim)
  clf = pumil(
      bags_train,
      metadata['train_lp'],
      metadata['train_up'] + metadata['train_un'],
      args)
  MI.print_evaluation_result(clf, bags_test, args)

#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
from sklearn import metrics


class Bag(object):
  def __init__(self, instances):
    self.instances = instances
    self.y = min(1, len(list(filter(lambda x: x['label'] == 1, instances)))) * 2 - 1
    self.unlabeled = False

  def __repr__(self):
    return "<Bag #data:{}, label:{}>".format(len(self.instances), self.y)

  def data(self):
    return np.array(list(map(lambda x: x['data'], self.instances)))

  def label(self):
    if self.unlabeled:
      return 0
    else:
      return self.y

  def mask(self):
    self.unlabeled = True

  def add_noise(self):
    m = np.max(list(map(lambda x: x['data'], self.instances)), axis=0)
    for i in range(len(self.instances)):
      z = np.random.normal(0, 0.01, m.shape[0])
      self.instances[i]['data'] += m * z

  def pca_reduction(self, pca):
    for i in range(len(self.instances)):
      self.instances[i]['data'] = pca.transform(self.instances[i]['data'].reshape(1, -1))[0]


def extract_bags(bags, Y, with_label = False):
  if with_label:
    return list(filter(lambda B: B.label() == Y, bags))
  else:
    return list(map(lambda B: B.data(), list(filter(lambda B: B.label() == Y, bags))))


def score(y, t, score_name):
  y      = np.array(y)
  t      = np.array(t)
  tp     = len(list(filter(lambda z: z == +1, y[np.where(t == +1)])))
  fp     = len(list(filter(lambda z: z == +1, y[np.where(t == -1)])))
  fn     = len(list(filter(lambda z: z == -1, y[np.where(t == +1)])))
  prec   = tp / (tp + fp) if (tp + fp) != 0 else 0
  recall = tp / (tp + fn) if (tp + fn) != 0 else 0

  if score_name == 'accuracy':
    return 1.0 * len(list(filter(lambda z: z[0] == z[1], zip(y, t)))) / len(y)

  elif score_name == 'precision':
    return prec

  elif score_name == 'recall':
    return recall

  elif score_name == 'f_score':
    return 2 * recall * prec / (recall + prec) if (recall + prec) != 0 else 0


def accuracy (y, t): return score(y, t, 'accuracy')
def precision(y, t): return score(y, t, 'precision')
def recall   (y, t): return score(y, t, 'recall')
def f_score  (y, t): return score(y, t, 'f_score')


def k_fold_cv(data, K):
  N = len(data)
  n = int(N / K)
  for k in range(K):
    yield data[:k*n] + data[(k+1)*n:], data[k*n:(k+1)*n]


def cross_validation(data, k):
  N = len(data)
  n = int(N / k)
  for t in range(k):
    vals = data[n*t:n*(t+1)]
    trains = np.r_[data[:n*t], data[n*(t+1):]]
    yield trains, vals


def print_evaluation_result(clf, bags_test, args):
  pred_score = np.array([clf(B.data()) for B in bags_test])
  pred_label = np.array([1 if score >= 0 else -1 for score in pred_score])
  true_label = np.array([B.y for B in bags_test])
  a = accuracy (pred_label, true_label)  # accuracy
  p = precision(pred_label, true_label)  # precision
  r = recall   (pred_label, true_label)  # recall
  f = f_score  (pred_label, true_label)  # F-score
  auc = metrics.roc_auc_score((true_label+1)/2, pred_score)

  if not args.aucplot:
    sys.stdout.write("""# accuracy,precision,recall,f-score,ROC-AUC
{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n""".format(a, p, r, f, auc))
    sys.stdout.flush()

  else:
    sys.stdout.write("""# accuracy,precision,recall,f-score,ROC-AUC
# {:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n""".format(a, p, r, f, auc))
    sys.stdout.flush()
    np.savetxt(sys.stdout.buffer, np.c_[pred_score, true_label])


from MI import PU
from MI import kernel
from MI import datasets

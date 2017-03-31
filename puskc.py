#!/usr/bin/env python
# coding: utf-8

import itertools
import argparse
import MI
import numpy as np

def train_pu_skc(data, args):
  reg = 1.0e-03
  bdim = len(data)

  theta = MI.PU.class_prior(data, args)
  basis = MI.kernel.minimax_basis(data, args.degree)
  model = MI.PU.SKC.train(data, basis, bdim, theta, reg, args)

  if args.verbose:
    print("#  estimated class prior = {:.6f}".format(theta))

  return model


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="PU-SKC experiment toolkit")

  parser.add_argument('--dataset',
      action   = 'store',
      required = True,
      type     = str,
      help     = 'multiple instance dataset')

  parser.add_argument('--prior',
      action   = 'store',
      default  = 0.1,
      type     = float,
      metavar  = '[0-1]',
      help     = 'class prior (the ratio of positive data)')

  parser.add_argument('--loss',
      action   = 'store',
      default  = 'double_hinge',
      type     = str,
      metavar  = ['double_hinge', 'squared'],
      help     = 'loss function')

  parser.add_argument('--degree',
      action   = 'store',
      default  = None,
      type     = str,
      metavar  = '[deg_lo|deg_hi|n_split]',
      help     = 'polynomial kernel degree (only effective for minimax basis)')

  parser.add_argument('-v', '--verbose',
      action   = 'store_true',
      default  = False,
      help     = 'verbose output')

  parser.add_argument('--aucplot',
      action   = 'store_true',
      default  = False,
      help     = 'output prediction score and true label for AUC plot')

  args = parser.parse_args()

  print("# {}".format('-'*80))
  print("# *** Experimental Setting ***")
  print("#   model                     : PU-SKC (loss function = {})".format(args.loss))
  print("#   basis                     : minimax (degree {})".format(args.degree))
#  print("#   validation                : 5 fold cross validation")
  print("# {}".format('-'*80))

  bags_train, bags_test, metadata = MI.datasets.load_dataset(args.dataset, args.prior)
  clf = train_pu_skc(bags_train, args)
  MI.print_evaluation_result(clf, bags_test, args)

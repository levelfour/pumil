#!/usr/bin/env python
# coding: utf-8

import MI
import numpy as np
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class ZeroOne(Function):
  """
  0-1 loss.
  """
  def forward(self, inputs):
    x = inputs[0]
    y = x < 0
    return utils.force_array(y, x.dtype),

  def backward(self, inputs, grad_outputs):
    x  = inputs[0]
    gy = grad_outputs[0]
    gx = (x == 0) * 0.5 * gy
    return utils.force_array(gx, x.dtype)


def dh_loss(z):
  """
  Double hinge loss function.

  l(z) = -z          if z <= -1
  l(z) = (1-z) / 2   if -1 < z <= 1
  l(z) = 0           if 1 < z
  """
  return F.relu(F.leaky_relu(-z-1, 0.5) + 1)


def ramp_loss(z):
  """
  Ramp loss function.

  l(z) = 1           if z <= -1
  l(z) = (1-z) / 2   if -1 < z <= 1
  l(z) = 0           if 1 < z
  """
  return F.hard_sigmoid(- 2.5 * z)


def zero_one_loss(z):
  """
  0-1 loss function.

  l(z) = 1           if z < 0
  l(z) = 0           if 0 <= z
  """
  return ZeroOne()(z)


def c_risk(theta, N1, N0, l):
  # (cf.) du Plessis et al. (2015)
  def _loss(y, t):
    if isinstance(y, np.ndarray) or isinstance(y, np.float64):
      y = y.astype(np.float32).reshape((1, 1))

    if N1 > 0 and N0 > 0:
      return F.sum((-theta/N1*y)*t + (l(-y)/N0)*(1-t))
    elif N0 == 0:
      return F.sum((-theta/N1*y)*t)
    elif N1 == 0:
      return F.sum((l(-y)/N0)*(1-t))

  return _loss


def nc_risk(theta, N1, N0, l):
  # (cf.) du Plessis et al. (2014)
  def _loss(y, t):
    if isinstance(y, np.ndarray) or isinstance(y, np.float64):
      y = y.astype(np.float32).reshape((1, 1))

    if N1 > 0 and N0 > 0:
      return F.sum((2*theta/N1*l(y))*t + (l(-y)/N0)*(1-t))
    elif N0 == 0:
      return F.sum((2*theta/N1*l(y))*t)
    elif N1 == 0:
      return F.sum((l(-y)/N0)*(1-t))

  return _loss


def prediction_error(bags, model, theta):
  N1 = len(MI.extract_bags(bags, 1))
  N0 = len(MI.extract_bags(bags, 0))
  error = nc_risk(theta, N1, N0, zero_one_loss)
  return sum(list(map(lambda B:
    float(error(model(B.data()), Variable(np.array([[B.label()]]).astype(np.float32))).data),
    bags))) - theta

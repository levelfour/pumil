import numpy as np
import MI

def r(x):
  return x.sum(axis=1).reshape(-1, 1)

def LSDD(P, Q, s, l):
  X = np.vstack((P, Q))
  d = X.shape[1]
  H = (np.pi * s**2)**(d/2) * np.exp(- (r(X**2) - 2*X.dot(X.T) + r(X**2).T) / (4*s**2))
  h = np.exp(- (r(X**2) - 2*X.dot(P.T) + r(P**2).T) / (2*s**2)).mean(axis=1) \
    - np.exp(- (r(X**2) - 2*X.dot(Q.T) + r(Q**2).T) / (2*s**2)).mean(axis=1)
  t = np.linalg.solve(H + l * np.eye(H.shape[0]), h)
  return t

def train(bags, width, reg, args):
  P = np.vstack(MI.extract_bags(bags, 1))
  Q = np.vstack(MI.extract_bags(bags, 0))

  t = LSDD(P, Q, width, reg)
  X = np.vstack((P, Q))

  def classifier(x):
    x = x.reshape(1, -1)
    return t.T.dot(np.exp(- (r(X**2) - 2*X.dot(x.T) + r(x**2).T) / (2*width**2)))

  return lambda X: np.max([classifier(x) for x in X])

def validation_error(validation_set, training_set, s, l, t):
  X = np.vstack((
    np.vstack(MI.extract_bags(training_set, 1)),
    np.vstack(MI.extract_bags(training_set, 0))))
  d = X.shape[1]
  P = np.vstack(MI.extract_bags(validation_set, 1))
  Q = np.vstack(MI.extract_bags(validation_set, 0))
  H = (np.pi * s**2)**(d/2) * np.exp(- (r(X**2) - 2*X.dot(X.T) + r(X**2).T) / (4*s**2))
  h = np.exp(- (r(X**2) - 2*X.dot(P.T) + r(P**2).T) / (2*s**2)).mean(axis=1) \
    - np.exp(- (r(X**2) - 2*X.dot(Q.T) + r(Q**2).T) / (2*s**2)).mean(axis=1)
  return t.dot(H.dot(t)) - 2*h.T.dot(t)

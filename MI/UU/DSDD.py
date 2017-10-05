import MI
import numpy as np
import cvxopt.solvers
from cvxopt import matrix
cvxopt.solvers.options['show_progress'] = False

def r(x):
  return x.sum(axis=1).reshape(-1, 1)

def train(bags, s, l, args):
  P = np.vstack(MI.extract_bags(bags, 0))
  Q = np.vstack(MI.extract_bags(bags, 1))

  n = len(P)
  m = len(Q)
  
  X = np.vstack((P, Q))
  KP = np.exp(-(r(P**2) - 2*P.dot(X.T) + r(X**2).T) / (2*s**2))
  KQ = np.exp(-(r(Q**2) - 2*Q.dot(X.T) + r(X**2).T) / (2*s**2))
  
  # initialization step
  L = np.r_[
    np.c_[l*np.eye(n+m), np.zeros((n+m, n)), np.zeros((n+m, m))],
    np.c_[np.zeros((n,n+m)), np.zeros((n, n)), np.zeros((n, m))],
    np.c_[np.zeros((m,n+m)), np.zeros((m, n)), np.zeros((m, m))],
  ]
  k = np.r_[
    np.zeros((n+m,1)),
    np.ones((n,1))/n,
    np.ones((m,1))/m,
  ]
  G = np.r_[
    np.c_[np.zeros((n,n+m)), -np.eye(n), np.zeros((n,m))],
    np.c_[KP, -np.eye(n), np.zeros((n,m))],
    np.c_[np.zeros((m,n+m)), np.zeros((m,n)), -np.eye(m)],
    np.c_[KQ, np.zeros((m,n)), -np.eye(m)],
  ]
  h = np.r_[
    np.zeros((n,1)),
    -np.ones((n,1)),
    np.zeros((m,1)),
    np.ones((m,1)),
  ]
  
  result = cvxopt.solvers.qp(matrix(L), matrix(k), matrix(G), matrix(h))
  a = np.array(result['x'])[:n+m]

  T = 10
  for t in range(T):
    # tighten the upper-bound
    b = KP.dot(a) >= 1
    c = KQ.dot(a) >= -1
    
    # minimize the upper-bound
    k = np.r_[
      -KP.T.dot(b)/n-KQ.T.dot(c)/m,
      np.ones((n,1))/n,
      np.ones((m,1))/m,
    ]

    result = cvxopt.solvers.qp(matrix(L), matrix(k), matrix(G), matrix(h))
    a = np.array(result['x'])[:n+m]
      
  def classifier(x):
    x = x.reshape(1, -1)
    return a.T.dot(np.exp(- (r(X**2) - 2*X.dot(x.T) + r(x**2).T) / (2*s**2)))
  
  return lambda X: np.max([classifier(x) for x in X])

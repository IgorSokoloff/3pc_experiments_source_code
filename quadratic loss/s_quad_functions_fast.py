"""
Auxillary functions for quadratic problem
sparse versions
"""

import numpy as np
import random
import time

from numpy.linalg import norm
from scipy import sparse
from scipy.optimize import minimize

#######################
# QUADRATIC FUNCTIONS #
#######################

sqnorm = lambda x: np.linalg.norm(x, ord=2)**2

def quadratic_loss(w, grad):
    #where grad = Ax - b (averaged per workers)
    return (0.5) * np.dot(w, grad)

def quadratic_grad_i(w, sX_i, y_i): #\nabla f_i, i.e. full grad per worker
    assert (y_i.dtype == 'float64')
    assert (sX_i.dtype == 'float64')
    assert (isinstance(sX_i, sparse.csr.csr_matrix))
    assert (isinstance(y_i, np.ndarray))
    assert (len(sX_i.shape) == 2)
    assert (len(y_i.shape) == 1)

    return sX_i.dot(w) - y_i

def quadratic_full_grad(w, sX, y):
    #sX - list of csr matrices
    #full_grad = np.zeros(w.shape, dtype=np.float32)
    #num_workers = X.shape[0]
    #for i in range(num_workers):
    #    full_grad += quadratic_grad_i(w, X[i], y[i])
    #return full_grad/num_workers
    assert (isinstance(sX[0], sparse.csr.csr_matrix))
    assert (isinstance(y, np.ndarray))
    assert (len(sX) == y.shape[0])

    meanS = sum(sX)/len(sX)
    return meanS.dot(w) - np.mean(y, axis=0)

def quadratic_hess_i(X_i):
    return X_i


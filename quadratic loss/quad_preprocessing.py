"""
syntheric data generation script
"""
import numpy as np
import time
import sys
import os
import argparse
from numpy.random import normal, uniform
from numpy.linalg import norm
import itertools
import pandas as pd
from matplotlib import pyplot as plt
import math
import datetime
from IPython import display
from scipy.optimize import minimize
from s_quad_functions_fast import *
from scipy import sparse
from scipy import linalg
from numpy.random import RandomState

def nan_check (lst):
    """
    Check whether has any item of list np.nan elements
    :param lst: list of datafiles (eg. numpy.ndarray)
    :return:
    """
    for i, item in enumerate (lst):
        if np.sum(np.isnan(item)) > 0:
            raise ValueError("nan files in item {0}".format(i))

myrepr = lambda x: repr(round(x, 4)).replace('.',',') if isinstance(x, float) else repr(x)
sqnorm = lambda x: norm(x, ord=2) ** 2

generate = 1

dataset = "synthetic"
loss_func = "quadratic"
dim = 1000
la = 1e-6

n_w_ar = np.array([10, 100, 1_000], dtype=int)
s_ar = np.array([0.05,0.1], dtype=float)

#test
#n_w_ar = np.array([10], dtype=int)
#s_ar = np.array([0.2], dtype=float)

user_dir = os.path.expanduser('~/')
project_path = os.getcwd() + "/"
data_path = project_path + "data_{0}/".format(dataset)

if not os.path.exists(data_path):
    os.mkdir(data_path)

if generate:
    print("dataset generation")
    for i, (n_w, s) in enumerate (itertools.product (n_w_ar, s_ar)):
        dataset_path = data_path + 'data_{0}_d{1}_nw{2}_s{3}/'.format(dataset, dim, n_w, myrepr(s))
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)

        rs = RandomState(12345)
        X = np.zeros((n_w, dim, dim), dtype=np.float64)
        y = np.zeros((n_w, dim), dtype=np.float64)

        #init matrices
        for i in range(n_w):
            xi_s = rs.normal(loc=0,scale=1, size=1)[0]
            nu_s = 1 + s*xi_s
            xi_b = rs.normal(loc=0,scale=1, size=1)[0]
            nu_b = s*xi_b

            X[i] = (nu_s/4)*((np.tri(dim, dim, 1, dtype=int) - np.tri(dim, dim, -2, dtype=int))*(-1) + 3*np.eye(dim))
            y_i = np.zeros(dim, dtype=np.float64)
            y_i[0] = (nu_s/4)*(-1 + nu_b)
            y[i] = y_i.copy()

        X_avg = np.mean(X, axis=0)
        la_min = linalg.eigh(a=X_avg, eigvals_only=True, turbo=True, type=1, eigvals=(0, 0))[0]

        #update matrices
        for i in range(n_w):
            X[i] += (la-la_min)*np.eye(dim)

        X_avg = np.mean(X, axis=0)
        X_sq_avg = sum(X[i]@X[i] for i in range(X.shape[0]))/X.shape[0]
        L_pm = np.sqrt(linalg.eigh(a=X_sq_avg - X_avg@X_avg, eigvals_only=True, turbo=True, type=1, eigvals=(dim-1, dim-1))[0])

        print(f"n_w: {n_w}; s: {s} L_pm: {L_pm}")
        np.save(dataset_path + 'L_pm_{0}_d{1}_nw{2}_s{3}'.format(dataset, dim, n_w, myrepr(s)), L_pm)
        sX = []
        np.save(dataset_path + 'y_{0}_d{1}_nw{2}_s{3}'.format(dataset, dim, n_w, myrepr(s)), y)

        for i in range(X.shape[0]):
            sX.append(sparse.csr_matrix(X[i]))
            assert(np.allclose(X[i], sX[i].toarray()))
            sparse.save_npz(dataset_path + 'sX-{4}_{0}_d{1}_nw{2}_s{3}'.format(dataset, dim, n_w, myrepr(s), i), sX[i])

    x_0 = np.zeros(dim, dtype=np.float64)
    x_0[0] = np.sqrt(dim)
    np.save(data_path + 'w_init_{0}_d{1}'.format(dataset, dim), x_0)


for i, (n_w, s) in enumerate (itertools.product (n_w_ar, s_ar)):
    print (n_w, s)

    dataset_path = data_path + 'data_{0}_d{1}_nw{2}_s{3}/'.format(dataset, dim, n_w, myrepr(s))
    L_0_path = dataset_path + 'L0_{0}_d{1}_nw{2}_s{3}'.format(dataset, dim, n_w, myrepr(s))
    L_path = dataset_path + 'L_{0}_d{1}_nw{2}_s{3}'.format(dataset, dim, n_w, myrepr(s))
    y_path = dataset_path + 'y_{0}_d{1}_nw{2}_s{3}.npy'.format(dataset, dim, n_w, myrepr(s))

    y = np.load(y_path)
    assert (y.dtype == 'float64')
    assert (len(y.shape) == 2)

    sX = []
    for i in range(n_w):
        sX.append(sparse.load_npz(dataset_path + 'sX-{4}_{0}_d{1}_nw{2}_s{3}.npz'.format(dataset, dim, n_w, myrepr(s), i)))
        assert (sX[i].dtype == 'float64')
        assert (len(sX[i].shape) == 2)

    print("L_0 computation")
    hess_f_0 = sum(sX)/n_w
    L_0 = np.real(sparse.linalg.eigs(A=hess_f_0, k=1, which="LR")[0][0])
    #L_0 = L_0.astype(np.float64)
    np.save(L_0_path, L_0)
    print(f"L_0: {L_0}")

    print("L computation")
    L = np.zeros(n_w, dtype=np.float64)
    for j in range(n_w):
        hess_f_j = sX[j]
        L[j] = np.real(sparse.linalg.eigs(A=hess_f_j, k=1, which="LR")[0][0])
    #L = L.astype(np.float64)
    np.save(L_path, L)
    print(f"L[:10]: {L[:10]}")

    x_0 = np.array(np.load(data_path + 'w_init_{0}_d{1}.npy'.format(dataset, dim)), dtype=np.float64)

    ##{
    x_star_path = dataset_path + 'x_star_{0}_d{1}_nw{2}_s{3}.npy'.format(dataset, dim, n_w, myrepr(s))
    f_star_path = dataset_path + 'f_star_{0}_d{1}_nw{2}_s{3}.npy'.format(dataset, dim, n_w, myrepr(s))

    print("optimization")

    grad = lambda w: quadratic_full_grad(w, sX, y)
    f = lambda w: quadratic_loss(w, grad(w))

    minimize_result = minimize(fun=f, x0=x_0, jac=grad, method="CG", tol=1e-16, options={"maxiter": 10000000})
    x_star, f_star = minimize_result.x, minimize_result.fun
    np.save(x_star_path, np.float64(x_star))
    np.save(f_star_path, np.float64(f_star))

    print(f"x_star[:10]: {x_star[:10]}")
    print(f"f_star: {f_star}")
##}

print ("finished!")
#test
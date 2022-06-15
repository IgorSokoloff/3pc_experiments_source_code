"""
3PCv2-RandK-Topk with full gradient
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
from scipy import sparse
from numpy.random import RandomState
from s_quad_functions_fast import *

myrepr = lambda x: repr(round(x, 4)).replace('.',',') if isinstance(x, float) else repr(x)
sqnorm = lambda x: norm(x, ord=2) ** 2

def stopping_criterion(sq_norm, eps, it, Nsteps, bits, Nbits):
    return (it <= Nsteps) and (sq_norm >=eps) and (bits <= Nbits)
        
def top_k_compressor(x, k):
    output = np.zeros(x.shape, dtype=np.float64)
    x_abs = np.abs(x)
    idx = np.argpartition(x_abs, -k)[-k:]  # Indices not sorted
    inds = idx[np.argsort(x_abs[idx])][::-1]
    output[inds] = x[inds]
    return output

def rand_k_compressor(x, k_rk, rs, probs):
    output = np.zeros(x.shape)
    d = x.shape[0]
    inds = rs.choice(a=np.arange(d), size=k_rk, replace=False, p=probs)
    output[inds] = x[inds]
    return (d/k_rk)* output

#upd: 09.01.2022
def compute_full_grads (A, x, b, n_workers):
    grad_ar = np.zeros((n_workers, x.shape[0]), dtype=np.float64)
    for i in range(n_workers):
        grad_ar[i] = quadratic_grad_i(x, A[i], b[i]).copy()
    return grad_ar

def anna_fg_estimator(A, x, b,  k_tk, k_rk, g_ar, grads_prev,  n_workers, rs, probs):
    #updated version using singe loop only
    grad = np.zeros(x.shape[0], dtype=np.float64)
    grads = np.zeros(grads_prev.shape, dtype=np.float64)
    for i in range(n_workers):
        grad_i = quadratic_grad_i(x, A[i], b[i]).copy()
        grads[i] = grad_i.copy()
        grad = grad + grad_i
        b_i = g_ar[i] + rand_k_compressor(grads[i] - grads_prev[i], k_rk, rs, probs)
        delta_i = grad_i - b_i
        g_ar[i] = b_i + top_k_compressor(delta_i, k_tk)
    size_value_sent = 64*1e-6
    grad = grad/n_workers
    return g_ar, size_value_sent, grads, grad

def anna_fg(x_0, x_star, f_star, A, b, stepsize, eps, k_tk, k_rk, n_workers, experiment_name, project_path, dataset, Nsteps, Nbits):
    currentDT = datetime.datetime.now()
    print(currentDT.strftime("%Y-%m-%d %H:%M:%S"))
    print(experiment_name + f" with Top{k_tk} and Rand{k_rk} finished")
    rs = RandomState(12345)
    dim = x_0.shape[0]
    probs = np.full(dim, 1 / dim)
    
    g_ar = compute_full_grads(A, x_0, b,  n_workers)
    grads = g_ar.copy()
    g = np.mean(g_ar, axis=0)

    f_x = quadratic_loss(x_0, g)
    sq_norm_ar = [sqnorm(g)]
    func_diff_ar = [f_x - f_star]

    bits_od_ar = [0]
    bits_bd_ar = [0]
    comms_ar = [0]
    arg_res_ar = [sqnorm(x_0 - x_star)] #argument residual \sqnorm{x^t - x_star}
    x = x_0.copy()
    it = 0
    PRINT_EVERY = 100000
    while stopping_criterion(sq_norm_ar[-1], eps, it, Nsteps, bits_od_ar[-1], Nbits):
        x = x - stepsize*g
        g_ar, size_value_sent, grads, grad = anna_fg_estimator(A, x, b, k_tk, k_rk, g_ar, grads, n_workers, rs, probs)
        g = np.mean(g_ar, axis=0)
        #f_x = quadratic_loss(x, grad)
        sq_norm_ar.append(sqnorm(grad))
        #func_diff_ar.append(f_x - f_star)

        it += 1
        bits_od_ar.append(it*(k_tk+k_rk)*size_value_sent)
        #bits_bd_ar.append(it*(dim+k)*size_value_sent)
        #arg_res_ar.append(sqnorm(x - x_star))
        comms_ar.append(it)
        if it%PRINT_EVERY ==0:
            display.clear_output(wait=True)
            #print(it, sq_norm_ar[-1])
            print_last_point_metrics(bits_od_ar, bits_bd_ar, comms_ar, comms_ar, arg_res_ar, func_diff_ar, sq_norm_ar)
            save_data(bits_od_ar, bits_bd_ar, comms_ar, comms_ar, arg_res_ar, func_diff_ar, sq_norm_ar, x.copy(), k_tk, k_rk, experiment_name, project_path, dataset)

    save_data(bits_od_ar, bits_bd_ar, comms_ar, comms_ar, arg_res_ar, func_diff_ar, sq_norm_ar, x.copy(), k_tk, k_rk, experiment_name, project_path, dataset)
    print(experiment_name + f" with Top{k_tk} and Rand{k_rk} finished")
    print("End-point:")
    print_last_point_metrics(bits_od_ar, bits_bd_ar, comms_ar, comms_ar, arg_res_ar, func_diff_ar, sq_norm_ar)
    #print("test: ", len(bits_od_ar), len(comms_ar), len(sq_norm_ar))

    #return bits_od_ar, bits_bd_ar, comms_ar, comms_ar, arg_res_ar, func_diff_ar, sq_norm_ar, x.copy()

def save_data(bits_od, bits_bd, epochs, comms, arg_res, func_diff, f_grad_norms, x_solution, k_tk, k_rk, experiment_name, project_path, dataset):
    print("data saving")
    experiment = '{0}_{1}_{2}'.format(experiment_name, k_tk, k_rk)
    logs_path = project_path + "logs/logs_{0}_{1}/".format(dataset, experiment)

    if not os.path.exists(project_path + "logs/"):
        os.makedirs(project_path + "logs/")

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    np.save(logs_path + 'iteration_bits_od' + '_' + experiment, np.array(bits_od, dtype=np.float64))
    #np.save(logs_path + 'iteration_bits_bd' + '_' + experiment, np.array(bits_bd, dtype=np.float64))
    #np.save(logs_path + 'iteration_epochs' + '_' +  experiment, np.array(epochs, dtype=np.float64))
    np.save(logs_path + 'iteration_comm' + '_' +    experiment, np.array(comms, dtype=np.float64))
    #np.save(logs_path + 'iteration_arg_res' + '_' + experiment, np.array(arg_res, dtype=np.float64))
    #np.save(logs_path + 'func_diff' + '_' +         experiment, np.array(func_diff, dtype=np.float64))
    np.save(logs_path + 'solution' + '_' +          experiment, x_solution)
    np.save(logs_path + 'norms' + '_' +             experiment, np.array(f_grad_norms, dtype=np.float64))

def print_last_point_metrics(bits_od, bits_bd, epochs, comms, arg_res, func_diff, f_grad_norms):
    print(f"comms: {comms[-1]}; bits_od: {bits_od[-1]}; arg_res: {arg_res[-1]}; f_grad_norms: {f_grad_norms[-1]}; func_diff: {func_diff[-1]}")

parser = argparse.ArgumentParser(description='Run top-k algorithm')
parser.add_argument('--max_it', action='store', dest='max_it', type=int, default=None, help='Maximum number of iteration')
parser.add_argument('--max_bits', action='store', dest='max_bits', type=float, default=None, help='Maximum number of bits transmitted from worker to server')
parser.add_argument('--k1', action='store', dest='k1', type=int, default=1, help='Sparcification parameter')
parser.add_argument('--k2', action='store', dest='k2', type=int, default=1, help='Sparcification parameter')
parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=1, help='Number of workers that will be used')
parser.add_argument('--factor', action='store', dest='factor', type=float, default=1, help='Stepsize factor')
parser.add_argument('--tol', action='store', dest='tol', type=float, default=1e-7, help='tolerance')
parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='synthetic', help='Dataset name for saving logs')
parser.add_argument('--scale', action='store', dest='s', type=float, default=0.2, help='Scale')
parser.add_argument('--dim', action='store', dest='dim', type=int, default=1000, help='Problem dimentionality')

args = parser.parse_args()

nsteps = args.max_it
nbits = args.max_bits
k_tk = args.k1
k_rk = args.k2
n_w = args.num_workers
dataset = args.dataset
loss_func = "quadratic"
factor = args.factor
eps = args.tol
s = args.s
dim = args.dim

'''
n_w = 10
k_tk = 5
k_rk = 5
nsteps = 200
nbits = nsteps*64*k_tk
dataset = "synthetic"
loss_func = "quadratic"
factor = 1.0
eps = 1e-8
s = 1.6
dim = 1000
'''

user_dir = os.path.expanduser('~/')
project_path = os.getcwd() + "/"
print(project_path)
data_path = project_path + "data_{0}/".format(dataset)

dataset_path = data_path + 'data_{0}_d{1}_nw{2}_s{3}/'.format(dataset, dim, n_w, myrepr(s))
L_0_path = dataset_path + 'L0_{0}_d{1}_nw{2}_s{3}.npy'.format(dataset, dim, n_w, myrepr(s))
L_path = dataset_path + 'L_{0}_d{1}_nw{2}_s{3}.npy'.format(dataset, dim, n_w, myrepr(s))
y_path = dataset_path + 'y_{0}_d{1}_nw{2}_s{3}.npy'.format(dataset, dim, n_w, myrepr(s))

y = np.load(y_path)
assert (y.dtype == 'float64')
assert (len(y.shape) == 2)

sX = []
for i in range(n_w):
    sX.append(sparse.load_npz(dataset_path + 'sX-{4}_{0}_d{1}_nw{2}_s{3}.npz'.format(dataset, dim, n_w, myrepr(s), i)))
    assert (sX[i].dtype == 'float64')
    assert (len(sX[i].shape) == 2)

if not os.path.isfile(L_0_path):
    print("WARNING: L_0 computation!")
    hess_f_0 = sum(sX)/n_w
    L_0 = np.real(sparse.linalg.eigs(A=hess_f_0, k=1, which="LR")[0][0])
    #L_0 = L_0.astype(np.float64)
    np.save(L_0_path, L_0)
else:
    L_0 = np.float64(np.load(L_0_path))

if not os.path.isfile(L_path):
    print("WARNING: L computation!")
    L = np.zeros(n_w, dtype=np.float64)
    for j in range(n_w):
        hess_f_j = sX[j]
        L[j] = np.real(sparse.linalg.eigs(A=hess_f_j, k=1, which="LR")[0][0])
    #L = L.astype(np.float64)
    np.save(L_path, L)
else:
    L = np.load(L_path)
    L = L.astype(np.float64)

x_0 = np.array(np.load(data_path + 'w_init_{0}_d{1}.npy'.format(dataset, dim)), dtype=np.float64)

##{
x_star_path = dataset_path + 'x_star_{0}_d{1}_nw{2}_s{3}.npy'.format(dataset, dim, n_w, myrepr(s))
f_star_path = dataset_path + 'f_star_{0}_d{1}_nw{2}_s{3}.npy'.format(dataset, dim, n_w, myrepr(s))

if (not os.path.isfile(x_star_path)) or (not os.path.isfile(f_star_path)):
    grad = lambda w: quadratic_full_grad(w, sX, y)
    f = lambda w: quadratic_loss(w, grad(w))

    minimize_result = minimize(fun=f, x0=x_0, jac=grad, method="CG", tol=1e-16, options={"maxiter": 10000000})
    x_star, f_star = minimize_result.x, minimize_result.fun
    np.save(x_star_path, np.float64(x_star))
    np.save(f_star_path, np.float64(f_star))
else:
    x_star = np.float64(np.load(x_star_path))
    f_star = np.float64(np.load(f_star_path))
##}


al_tk = k_tk/dim
omega = dim/k_rk - 1
theta = al_tk
beta = (1 - al_tk)*omega
Lt = np.sqrt(np.mean(L**2))
step_size_anna_fg = np.float64((1/(L_0 + Lt*np.sqrt(beta/theta)))*factor)

experiment_name = "anna-randk-topk-fg_d{0}_nw{1}_s{2}_{3}x".format(dim, n_w, myrepr(s), myrepr(factor))

print ('------------------------------------------------------------------------------------------------')
start_time = time.time()
anna_fg(x_0, x_star, f_star, sX, y, step_size_anna_fg, eps,  k_tk, k_rk, n_w, experiment_name, project_path, dataset, Nsteps=nsteps, Nbits=nbits)
time_diff = time.time() - start_time
print(f"Computation time: {time_diff} sec")
print ('------------------------------------------------------------------------------------------------')



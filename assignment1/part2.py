import numpy as np
import math
from math import log
import random

""" The following contains all the functions used for part 2 of the assignment, without 
The code for testing"""

# part 2.1
def generateDataset(N, f, sigma):
    mu = 0
    s = np.array(np.random.normal(mu, sigma, N))
    x = np.array(np.linspace(0.0, 1.0, N))
    vf = np.vectorize(f)
    t = np.add(vf(x), s)
    return (x, t)

# part 2.2
def OptimizeLS(x, t, M):
    phi = np.vstack([np.power(x_i, m) for m in range(M)] for x_i in x)
    prod = np.dot(phi.T, phi)
    i = np.linalg.inv(prod)
    phi_mults = np.dot(i, phi.T)
    w = np.dot(phi_mults, t)
    return w

# part 2.3
def optimizePLS(x, t, M, l):
    phi = np.vstack([np.power(x_i, m) for m in range(M)] for x_i in x)
    prod = np.dot(phi.T, phi)
    l_i = np.dot(l, np.identity(prod.shape[0]))
    fixed = np.add(prod, l_i)
    i = np.linalg.inv(fixed)
    phi_mults = np.dot(i, phi.T)
    w = np.dot(phi_mults, t)
    return w

#part 2.4
def generateDataset3(N, f, sigma):
    mu = 0
    s = np.array(np.random.normal(mu, sigma, N))
    
    x_1 = np.array(np.linspace(0.0, 1.0, N))
    x_2 = np.copy(x_1)
    x_3 = np.copy(x_1)
    np.random.shuffle(x_2)
    np.random.shuffle(x_3)
    
    vf = np.vectorize(f)
    
    t_1 = np.add(vf(x_1), s)
    t_2 = np.add(vf(x_2), s)
    t_3 = np.add(vf(x_3), s)
    
    return [(x_1, t_1), (x_2, t_2), (x_3, t_3)]

def N_E(x, t, w):
    err = 0.0
    for i in range(len(t)):
        poly = 0.0
        for m in range(len(w)):
            poly += (w[m] * (x[i] ** m))
        err += (t[i] - poly) ** 2
    err = err ** 0.5
    err = (1 / float(len(t))) * err
    return err

def optimizePLSLambda(xt, tt, xv, tv, M):
    w_max = np.zeros(M)
    avg_err = 1.0
    for i in range(-40, -20):
        l = np.exp(i)
        w_curr = optimizePLS(xt, tt, M, l)
        curr_err = N_E(xv, tv, w_curr)
        if(curr_err < avg_err):
            avg_err = curr_err
            w_max = np.copy(w_curr)
    return w_max

def bayesianEstimator(x, t, M, alpha, sigma2):
    N = len(x)
    
    #builds phi for given vector xx
    def phi(xx):
        return np.array([(xx ** i) for i in range(M + 1)])
    
    #calculate S
    alpha_i = alpha * np.eye(M + 1)
    S = np.zeros((M + 1, M + 1))
    for i in range(N):
        phi_xi = phi(x[i])
        S += np.outer(phi_xi, phi_xi.T)
    S = np.linalg.inv(alpha_i + (S / sigma2))
    
    #calculate m(x)
    def m(xx):
        phi_t = phi(xx).T
        xt_sum = np.zeros(M + 1)
        for i in range(N):
            xt_sum += phi(x[i]) * t[i]
        return (1 / sigma2) * np.dot(np.dot(phi_t, S), xt_sum)
    
    #calculate s2(xx)
    def var(xx):
        phi_x = phi(xx)
        return sigma2 + np.dot(phi_x.T, np.dot(S, phi_x))
    
    mean = lambda x_t: m(x_t)
    variance = lambda x_t: var(x_t)
    
    return (mean, variance)
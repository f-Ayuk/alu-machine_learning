#!/usr/bin/env python3
"""
Calculates the Bayesian Information Criterion for a GMM.
"""

import numpy as np expectation_maximization = import('8-EM').expectation_maximization

def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, 
        verbose=False):
    """
    Find the best number of clusters for a GMM using BIC
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
    return None, None, None, None

n, d = X.shape
if n < 2 or d < 1:
    return None, None, None, None

if kmax is None:
    kmax = n

if not isinstance(kmin, int) or isinstance(kmin, bool) or kmin < 1:
    return None, None, None, None

if not isinstance(kmax, int) or isinstance(kmax, bool) or kmax < 1:
    return None, None, None, None

if kmax < kmin:
    return None, None, None, None

if not isinstance(iterations, int) or isinstance(iterations, bool):
    return None, None, None, None
if iterations < 1:
    return None, None, None, None

if not isinstance(tol, (int, float)) or isinstance(tol, bool):
    return None, None, None, None
if not np.isfinite(tol) or tol < 0:
    return None, None, None, None

if not isinstance(verbose, bool):
    return None, None, None, None

l = []
b = []
best_k = None
best_result = None
best_bic = np.inf

for k in range(kmin, kmax + 1):
    result = expectation_maximization(
        X, k, iterations, tol, verbose
    )

    if result[0] is None:
        return None, None, None, None

    pi, m, S, g, log_likelihood = result

    # Number of free parameters in a GMM:
    # k - 1 mixing weights, k*d means, and
    # k*d*(d+1)/2 covariance parameters.
    parameters = (
        (k - 1) + k * d + k * d * (d + 1) // 2
    )

    bic = parameters * np.log(n) - 2 * log_likelihood

    l.append(log_likelihood)
    b.append(bic)

    if bic < best_bic:
        best_bic = bic
        best_k = k
        best_result = (pi, m, S)

return best_k, best_result, np.array(l), np.array(b)

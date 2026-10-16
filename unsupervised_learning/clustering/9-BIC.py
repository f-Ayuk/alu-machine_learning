#!/usr/bin/env python3
"""
Function finds the best number of clusters for a GMM using
the Bayesian Information Criterion (BIC)
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


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

    if not isinstance(kmin, int) or kmin < 1 or kmin >= n:
        return None, None, None, None

    if kmax is None:
        kmax = n - 1
    elif not isinstance(kmax, int) or kmax < 2 or kmax >= n:
        return None, None, None, None

    if kmax < kmin:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    ks = np.arange(kmin, kmax + 1)
    log_likelihoods = np.zeros(len(ks))
    bics = np.zeros(len(ks))
    best_bic = np.inf
    best_k = None
    best_result = None

    for i, k in enumerate(ks):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if log_likelihood is None:
            return None, None, None, None

        log_likelihoods[i] = log_likelihood

        p = (k - 1) + k * d + k * d * (d + 1) / 2
        bics[i] = p * np.log(n) - 2 * log_likelihood

        if bics[i] < best_bic:
            best_bic = bics[i]
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, log_likelihoods, bics

#!/usr/bin/env python3
"""Determines the best number of clusters for a GMM using BIC."""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5,
        verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    n, d = X.shape

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None

    if kmax < kmin:
        return None, None, None, None

    if kmax >= n:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    likelihoods = []
    bics = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None, None

        parameters = (k - 1) + k * d + k * d * (d + 1) / 2
        bic = parameters * np.log(n) - 2 * log_likelihood

        results.append((pi, m, S))
        likelihoods.append(log_likelihood)
        bics.append(bic)

    likelihoods = np.array(likelihoods)
    bics = np.array(bics)

    best_index = np.argmin(bics)
    best_k = best_index + kmin
    best_result = results[best_index]

    return best_k, best_result, likelihoods, bics

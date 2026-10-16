#!/usr/bin/env python3
"""
Find the optimum number of clusters by calculating variance.
"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests different values of k to determine the optimum number
    of clusters based on the variance.

    Args:
        X (numpy.ndarray): Dataset of shape (n_samples, n_features).
        kmin (int): Minimum number of clusters.
        kmax (int): Maximum number of clusters.
        iterations (int): Maximum number of iterations for K-means.

    Returns:
        tuple: A list of (centroids, cluster assignments) and
        a list of variance differences, or (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]
    elif not isinstance(kmax, int) or kmax <= 0:
        return None, None

    if kmax < kmin:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    d_vars = []
    previous_var = None

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))

        current_var = variance(X, C)

        if previous_var is not None:
            d_vars.append(previous_var - current_var)

        previous_var = current_var

    return results, d_vars

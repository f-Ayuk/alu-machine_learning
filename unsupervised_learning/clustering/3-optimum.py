#!/usr/bin/env python3
"""
Find the optimum number of clusters based on variance.
"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Test different values of k to find the optimum number of clusters.

    Args:
        X (numpy.ndarray): Dataset of shape (n_samples, n_features).
        kmin (int): Minimum number of clusters.
        kmax (int): Maximum number of clusters.
        iterations (int): Maximum number of K-means iterations.

    Returns:
        tuple: A list of clustering results and a list of variance
        differences, or (None, None) if an argument is invalid.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if type(kmin) is not int or kmin <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]
    elif type(kmax) is not int or kmax <= 0:
        return None, None

    if kmax < kmin:
        return None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None

    results = []
    d_vars = []
    previous_var = None

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))

        current_var = variance(X, C)

        if previous_var is None:
            d_vars.append(current_var)
        else:
            d_vars.append(previous_var - current_var)

        previous_var = current_var

    return results, d_vars

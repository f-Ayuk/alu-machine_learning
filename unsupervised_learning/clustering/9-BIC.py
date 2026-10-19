
#!/usr/bin/env python3
"""
Determines the best number of clusters for a GMM using BIC.
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5,
        verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n - 1

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None

    if kmax < kmin:
        return None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None

    if not isinstance(verbose, bool):
        return None, None, None

    if kmax >= n:
        return None, None, None

    results = []
    l = []
    b = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None

        num_params = (k - 1) + k * d + k * d * (d + 1) / 2
        bic = num_params * np.log(n) - 2 * log_likelihood

        results.append((pi, m, S))
        l.append(log_likelihood)
        b.append(bic)

    l = np.array(l)
    b = np.array(b)

    best_k = np.argmin(b) + kmin
    best = results[best_k - kmin]

    return best_k, best, l, b

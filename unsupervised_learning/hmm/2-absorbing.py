#!/usr/bin/env python3
"""
Determines if a Markov chain is absorbing
"""
import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing.

    :param P: square 2D numpy.ndarray of shape (n, n)
              representing the transition matrix
    :return: True if it is absorbing, or False on failure
    """
    if type(P) is not np.ndarray:
        return False
    if P.ndim != 2:
        return False

    n, n_t = P.shape
    if n != n_t or n == 0:
        return False

    if not np.all(np.isfinite(P)) or np.any(P < 0):
        return False

    if not np.allclose(np.sum(P, axis=1), 1):
        return False

    diagonal = np.diag(P)
    absorb = np.isclose(diagonal, 1)

    if not np.any(absorb):
        return False

    reachable = absorb.copy()

    while True:
        previous = reachable.copy()

        for row in range(n):
            for col in range(n):
                if P[row, col] > 0 and reachable[col]:
                    reachable[row] = True

        if np.array_equal(previous, reachable):
            break

    return bool(np.all(reachable))

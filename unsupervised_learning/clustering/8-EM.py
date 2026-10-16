#!/usr/bin/env python3
'''
Expectation Maximization for a Gaussian Mixture Model
'''

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    '''
    Expectation maximization for a GMM
    '''
    if not isinstance(X, np.ndarray):
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, log_likelihood = expectation(X, pi, m, S)

    if verbose:
        print("Log Likelihood after 0 iterations: {:.5f}".format(
            log_likelihood))

    for i in range(iterations):
        pi, m, S = maximization(X, g)
        g, new_log_likelihood = expectation(X, pi, m, S)

        converged = abs(new_log_likelihood - log_likelihood) <= tol

        if verbose and ((i + 1) % 10 == 0 or converged):
            print("Log Likelihood after {} iterations: {:.5f}".format(
                i + 1, new_log_likelihood))

        log_likelihood = new_log_likelihood

        if converged:
            break

    return pi, m, S, g, log_likelihood

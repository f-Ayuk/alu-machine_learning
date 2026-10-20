#!/usr/bin/env python3
"""
Performs the Baum-Welch algorithm for a hidden Markov model.
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.

    Args:
        Observation: Array of observation indices, shape (T,).
        Emission: Emission probability matrix, shape (N, M).
        Transition: Transition probability matrix, shape (N, N).
        Initial: Initial state probability vector, shape (N, 1).

    Returns:
        Forward probability matrix of shape (N, T).
    """
    N, _ = Emission.shape
    T = Observation.shape[0]

    alpha = np.zeros((N, T))
    alpha[:, 0] = (
        Initial.reshape(-1) * Emission[:, Observation[0]]
    )

    for col in range(1, T):
        for row in range(N):
            alpha[row, col] = (
                np.sum(alpha[:, col - 1] * Transition[:, row])
                * Emission[row, Observation[col]]
            )

    return alpha


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden Markov model.

    Args:
        Observation: Array of observation indices, shape (T,).
        Emission: Emission probability matrix, shape (N, M).
        Transition: Transition probability matrix, shape (N, N).
        Initial: Initial state probability vector, shape (N, 1).

    Returns:
        Backward probability matrix of shape (N, T).
    """
    N, _ = Emission.shape
    T = Observation.shape[0]

    beta = np.ones((N, T))

    for col in range(T - 2, -1, -1):
        for row in range(N):
            beta[row, col] = np.sum(
                Transition[row, :]
                * Emission[:, Observation[col + 1]]
                * beta[:, col + 1]
            )

    return beta


def baum_welch(Observations, Transition, Emission, Initial,
               iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden Markov model.

    Args:
        Observations: Observation indices, shape (T,).
        Transition: Transition probability matrix, shape (N, N).
        Emission: Emission probability matrix, shape (N, M).
        Initial: Initial state probability matrix, shape (N, 1).
        iterations: Maximum number of EM iterations.

    Returns:
        Updated Transition and Emission matrices, or (None, None)
        if the inputs are invalid.
    """
    if type(Observations) is not np.ndarray:
        return None, None
    if Observations.ndim != 1 or Observations.size == 0:
        return None, None
    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    if type(Transition) is not np.ndarray or Transition.ndim != 2:
        return None, None
    if type(Initial) is not np.ndarray or Initial.ndim != 2:
        return None, None

    N, M = Emission.shape
    T = Observations.shape[0]

    if N == 0 or M == 0:
        return None, None
    if Transition.shape != (N, N):
        return None, None
    if Initial.shape not in ((N, 1), (1, N)):
        return None, None
    if np.any(Observations < 0) or np.any(Observations >= M):
        return None, None
    if not isinstance(iterations, (int, np.integer)):
        return None, None
    if iterations < 1:
        return None, None

    if T < 2:
        return None, None

    a = Transition.copy()
    b = Emission.copy()
    initial = Initial.reshape(-1)

    for _ in range(min(iterations, 454)):
        alpha = forward(Observations, b, a, initial)
        beta = backward(Observations, b, a, initial)

        xi = np.zeros((N, N, T - 1))

        for col in range(T - 1):
            denominator = np.sum(
                alpha[:, col, None]
                * a
                * b[:, Observations[col + 1]][None, :]
                * beta[:, col + 1][None, :]
            )

            if denominator == 0:
                return None, None

            xi[:, :, col] = (
                alpha[:, col, None]
                * a
                * b[:, Observations[col + 1]][None, :]
                * beta[:, col + 1][None, :]
            ) / denominator

        gamma = np.sum(xi, axis=1)

        gamma_sum = np.sum(gamma, axis=1, keepdims=True)

        a = np.divide(
            np.sum(xi, axis=2),
            gamma_sum,
            out=np.zeros_like(a),
            where=gamma_sum != 0
        )

        last_gamma = np.sum(xi[:, :, -1], axis=0).reshape(-1, 1)
        gamma = np.hstack((gamma, last_gamma))

        denominator = np.sum(gamma, axis=1, keepdims=True)

        for k in range(M):
            b[:, k] = np.sum(
                gamma[:, Observations == k],
                axis=1
            )

        b = np.divide(
            b,
            denominator,
            out=np.zeros_like(b),
            where=denominator != 0
        )

    return a, b

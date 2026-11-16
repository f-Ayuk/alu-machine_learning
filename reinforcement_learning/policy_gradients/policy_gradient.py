#!/usr/bin/env python3
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight matrix.

    Args:
        matrix: State matrix (m x n)
        weight: Weight matrix (n x actions)

    Returns:
        The action probabilities using softmax.
    """
    z = np.matmul(matrix, weight)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

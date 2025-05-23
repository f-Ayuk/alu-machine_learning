#!/usr/bin/env python3
"""Getting the transpose of a matrix"""


def matrix_transpose(matrix):
    """
    Calculates the transpose of a 2D matrix.

    Args:
        matrix: The input matrix.

    Returns:
        A new matrix representing the transpose of the input matrix.
    """
    rows = len(matrix)
    cols = len(matrix[0])  # Assuming all rows have the same number of columns
    transposed = [[0 for _ in range(rows)] for _ in range(cols)]
    # Initialize empty matrix

    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]  # Swap elements

    return transposed

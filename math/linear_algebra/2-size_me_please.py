#!/usr/bin/env python3

def matrix_shape(matrix):
     """
    Calculates the shape of a matrix.

    Args:
        matrix: The input matrix.

    Returns:
        A list of integers representing the shape of the matrix.
    """
    shape = []
    while type(matrix) is list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape

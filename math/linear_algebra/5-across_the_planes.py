#!/usr/bin/env python3
def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None  # Matrices have different shapes
    rows, cols = len(mat1), len(mat1[0])
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = mat1[i][j] + mat2[i][j]
    return result

#!/usr/bin/env python3
"""Getting the cofactor"""


def determinant(matrix):
    """Calculates the determinant of a matrix."""
    if not all(isinstance(row, list) for row in matrix) or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")
    if n == 0:
        return 1
        # The determinant of a 0x0 matrix is conventionally defined as 1
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for c in range(n):
        sub_matrix = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(sub_matrix)
    return det
def minor(matrix):
    """Calculates the minor matrix of a matrix."""
    if not isinstance(matrix, list) or not all(isinstance(row, list)
        for row in matrix):
            raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:]
            for row in (matrix[:i] + matrix[i+1:])]
                minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)
    return minor_matrix
def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix."""
    if not isinstance(matrix, list) or not all(isinstance(row, list) 
        for row in matrix):
            raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    cofactor_matrix = minor(matrix)
    for i in range(n):
        for j in range(n):
            cofactor_matrix[i][j] *= (-1) ** (i + j)
    return cofactor_matrix

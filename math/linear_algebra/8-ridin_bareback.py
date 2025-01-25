#!/usr/bin/env python3
def mat_mul(mat1, mat2):
  """
  Performs matrix multiplication between two 2D matrices.

  Args:
      mat1: The first input matrix.
      mat2: The second input matrix.

  Returns:
      A new matrix containing the result of matrix multiplication, or None if
      the matrices cannot be multiplied.
  """
  # Check if inner dimensions are compatible for multiplication
    if len(mat1[0]) != len(mat2):
      return None  # Cannot multiply matrices with incompatible inner dimensions
    rows1, cols1 = len(mat1), len(mat1[0])
    rows2, cols2 = len(mat2), len(mat2[0])
    result = [[0 for _ in range(cols2)] for _ in range(rows1)]
    # Perform element-wise multiplication and summation
    for i in range(rows1):
      for j in range(cols2):
        for k in range(cols1):
          result[i][j] += mat1[i][k] * mat2[k][j]
    return result

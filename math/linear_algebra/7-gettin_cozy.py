#!/usr/bin/env python3
def cat_matrices2D(mat1, mat2, axis=0):
  """
  Concatenates two 2D matrices along a specific axis.
  """
  def cat_matrices2D(matrix1, matrix2, axis=0):
    """Concatenating two matrices along axis 0"""
    if axis == 0:
        if len(matrix1[0]) != len(matrix2[0]):
            return None
        return matrix1 + matrix2
    if axis == 1:
        if len(matrix1) != len(matrix2):
            return None
        return [matrix1[i] + matrix2[i] for i in range(len(matrix1))]
    return None

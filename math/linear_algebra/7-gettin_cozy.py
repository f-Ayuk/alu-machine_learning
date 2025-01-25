#!/usr/bin/env python3
def cat_matrices2D(mat1, mat2, axis=0):
  """
  Concatenates two 2D matrices along a specific axis.
  """
  def cat_matrices2D(mat1, mat2, axis=0):
    if axis == 0:
        if any(len(row) != len(mat2[0]) for row in mat1) or any(len(row) != len(mat1[0]) for row in mat2):
            return None
        return [row.copy() for row in mat1] + [row.copy() for row in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [r1 + r2 for r1, r2 in zip(mat1, mat2)]
    else:
        return None

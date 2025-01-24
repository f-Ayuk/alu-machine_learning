#!/usr/bin/env python3
def cat_matrices2D(mat1, mat2, axis=0):
  """
  Concatenates two 2D matrices along a specific axis.
  
  Args:
      mat1: The first input matrix.
      mat2: The second input matrix.
      axis: The axis (0 for rows, 1 for columns) along which to concatenate.
          Defaults to 0 (rows).

  Returns:
      A new matrix containing the concatenated matrices, or None if
      concatenation is not possible.
  """
  if axis == 0:
    # Concatenate along rows
    if len(mat2[0]) != len(mat1[0]):
      return None  # Different column lengths, cannot concatenate
    return mat1 + mat2
  elif axis == 1:
    # Concatenate along columns
    if len(mat1) != len(mat2):
      return None  # Different row lengths, cannot concatenate
    return [[row[i] for row in (mat1 + mat2)] for i in range(len(mat1[0]))]
  else:
    return None  # Invalid axis value

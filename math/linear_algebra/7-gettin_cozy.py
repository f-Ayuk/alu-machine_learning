#!/usr/bin/env python3
def cat_matrices2D(mat1, mat2, axis=0):
  """
  Concatenates two 2D matrices along a specific axis.
  """
  if axis == 0:
    # Concatenate along rows
    if len(mat1[0]) != len(mat2[0]):
      return None  # Different column lengths, cannot concatenate
    return [row + col for row, col in zip(mat1, mat2)]
  elif axis == 1:
    # Concatenate along columns
    if len(mat1) != len(mat2):
      return None  # Different row lengths, cannot concatenate
    return [[row[i] for row in (mat1 + mat2)] for i in range(len(mat1[0]))]
  else:
    return None  # Invalid axis value

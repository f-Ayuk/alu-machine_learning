#!/usr/bin/env python3
mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6]]
mat3 = [[7], [8]]
mat4 = cat_matrices2D(mat1, mat2)
mat5 = cat_matrices2D(mat1, mat3, axis=1)
print(mat4)  # Output: [[1, 2], [3, 4], [5, 6]]
print(mat5)  # Output: [[1, 2, 7], [3, 4, 8]]

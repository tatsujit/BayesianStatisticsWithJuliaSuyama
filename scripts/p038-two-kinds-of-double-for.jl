[(x, y) for x in 1:3, y in 1:2]
# 3×2 Matrix{Tuple{Int64, Int64}}:
#  (1, 1)  (1, 2)
#  (2, 1)  (2, 2)
#  (3, 1)  (3, 2)

[(x, y) for x in 1:3 for y in 1:2]
# 6-element Vector{Tuple{Int64, Int64}}:
#  (1, 1)
#  (1, 2)
#  (2, 1)
#  (2, 2)
#  (3, 1)
#  (3, 2)


# Explanation by Copilot:
# The first comprehension creates a 2D array (matrix) where each element is a tuple (x, y) for the corresponding indices.
# The second comprehension creates a 1D array (vector) by iterating over all combinations of x and y, resulting in a flat list of tuples.
#
# In summary, the difference lies in the structure of the output: the first is a matrix, while the second is a flat vector.
# This distinction is important when you want to control the shape of the resulting collection in Julia.

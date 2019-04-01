import math

import numpy as np

"""
Fixes a vector to a given size.
If len(vector) > size, the vector is cut.
If len(vector) < size, the vector is repeated until it matches the desired size.
"""


def size_control(vector, size):
    if len(vector) < size:
        vector *= math.ceil(size / len(vector))
    return vector[:size]

def multi_size_control(vectors, size):
    sized_vectors = np.zeros([len(vectors), size])
    for i, vector in enumerate(vectors):
        sized_vectors[i] = size_control(vector, size)
    return sized_vectors
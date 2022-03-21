from __future__ import division
import math

from functools import reduce


def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors):
    return reduce(vector_add, vectors)


def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols


def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)


def get_column(A, j):
    return [A_i[j] for A_i in A]


def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i, j) for j in range(num_cols)]
            for i in range(num_rows)]


def vector_subtract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def scalar_multiply(c, v):
    return [c * v_i for v_i in v]


# this isn't right if you don't from __future__ import division
def vector_mean(vectors):
    """compute the vector whose i-th element is the mean of the
    i-th elements of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))


def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))


def distance(v, w):
    return math.sqrt(squared_distance(v, w))

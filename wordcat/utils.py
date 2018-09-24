"""
Contains all classes and functions that provide utility of some kind, such as
debug printing capability and sparse matrix equality.
"""
import numpy as np


def sparse_equal(A, B):
    """
    Compares the two specified sparse matrices for numeric equality,
    returning true if they have the same dimensions and match numerically for
    every possible element (zero or otherwise).

    :param A: A sparse matrix to compare.
    :param B: Another sparse matrix to compare.
    :return: Whether or not two sparse matrices are numerically equaal.
    """
    return np.array_equal(A.data, B.data) and \
           np.array_equal(A.indices, B.indices) and \
           np.array_equal(A.indptr, B.indptr)

"""
Contains unit tests to ensure that all non-specific utility functions work as
expected.
"""
import numpy as np
from unittest import TestCase

from scipy.sparse import csr_matrix

from wordcat.utils import sparse_equal


class UtilsTest(TestCase):
    """
    Test suite for the "utils" package.
    """

    def test_sparse_equals_using_different_matrices(self):
        A = csr_matrix(np.array([[0, 0, 1, 0], [2, 0, 0, 0], [0, 1, 0, 0]]))
        B = csr_matrix(np.array([[0, 0, 2, 0], [2, 0, 0, 0], [0, 1, 0, 0]]))

        self.assertFalse(sparse_equal(A, B))

    def test_sparse_equals_using_identical_matrices(self):
        A = csr_matrix(np.array([[0, 0, 1, 0], [2, 0, 0, 0], [0, 1, 0, 0]]))
        B = csr_matrix(np.array([[0, 0, 1, 0], [2, 0, 0, 0], [0, 1, 0, 0]]))

        self.assertTrue(sparse_equal(A, B))

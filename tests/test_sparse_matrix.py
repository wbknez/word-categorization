"""
Contains unit tests to ensure that the sparse matrix data structure works as
intended.
"""
import numpy as np
from unittest import TestCase

from wordcat.sparse import SparseMatrix


class SparseMatrixTest(TestCase):
    """
    Test suite for SparseMatrix.
    """

    @staticmethod
    def make_sparse(dense_matrix):
        cols = []
        data = []
        rows = []

        for row_index, row in enumerate(dense_matrix):
            for col_index, col in enumerate(row):
                if col != 0:
                    cols.append(col_index)
                    data.append(col)
                    rows.append(row_index)

        return SparseMatrix(np.array(data), np.array(rows, np.int),
                            np.array(cols, np.int),
                            (len(dense_matrix), len(dense_matrix[0])))

    def test_init_throws_when_col_dtype_is_not_integral(self):
        with self.assertRaises(ValueError):
            SparseMatrix(np.zeros(10), np.zeros(10, np.int),
                         np.zeros(10, np.float), (10, 10))

    def test_init_throws_when_data_and_column_dimensions_are_unequal(self):
        with self.assertRaises(ValueError):
            SparseMatrix(np.zeros(8), np.zeros(10, np.int),
                         np.zeros(10, np.int), (10, 10))

    def test_init_throws_when_row_and_column_dimensions_are_unequal(self):
        with self.assertRaises(ValueError):
            SparseMatrix(np.zeros(10), np.zeros(10, np.int),
                         np.zeros(8, np.int), (10, 10))

    def test_init_throws_when_row_dtype_is_not_integral(self):
        with self.assertRaises(ValueError):
            SparseMatrix(np.zeros(10), np.zeros(10, np.float),
                         np.zeros(10, np.int), (10, 10))

    def test_get_data_column_using_simple_matrix(self):
        mat = SparseMatrixTest.make_sparse([[3, 0, 0], [0, 0, 1], [5, 0, 3]])

        expected0 = np.array([3, 5])
        expected1 = np.array([])
        expected2 = np.array([1, 3])

        result0 = mat.get_data_column(0)
        result1 = mat.get_data_column(1)
        result2 = mat.get_data_column(2)

        self.assertTrue(np.array_equal(result0, expected0))
        self.assertTrue(np.array_equal(result1, expected1))
        self.assertTrue(np.array_equal(result2, expected2))

    def test_get_data_row_using_simple_matrix(self):
        mat = SparseMatrixTest.make_sparse([[3, 0, 0], [0, 0, 1], [5, 0, 3]])

        expected0 = np.array([3])
        expected1 = np.array([1])
        expected2 = np.array([5, 3])

        result0 = mat.get_data_row(0)
        result1 = mat.get_data_row(1)
        result2 = mat.get_data_row(2)

        self.assertTrue(np.array_equal(result0, expected0))
        self.assertTrue(np.array_equal(result1, expected1))
        self.assertTrue(np.array_equal(result2, expected2))

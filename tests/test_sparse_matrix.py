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

    def test_init_throws_when_col_dtype_is_not_integral(self):
        with self.assertRaises(ValueError):
            SparseMatrix(np.zeros(10), np.zeros(10, np.int),
                         np.zeros(10, np.float), (10, 10))

    def test_init_throws_when_and_column_dimensions_are_unequal(self):
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

    def test_get_column_using_simple_matrix(self):
        mat = SparseMatrix.from_list([[3, 0, 0], [0, 0, 1], [5, 0, 3]])

        expected0 = np.array([3, 5])
        expected1 = np.array([])
        expected2 = np.array([1, 3])

        result0 = mat.get_column(0)
        result1 = mat.get_column(1)
        result2 = mat.get_column(2)

        self.assertTrue(np.array_equal(result0, expected0))
        self.assertTrue(np.array_equal(result1, expected1))
        self.assertTrue(np.array_equal(result2, expected2))

    def test_get_row_using_simple_matrix(self):
        mat = SparseMatrix.from_list([[3, 0, 0], [0, 0, 1], [5, 0, 3]])

        expected0 = np.array([3])
        expected1 = np.array([1])
        expected2 = np.array([5, 3])

        result0 = mat.get_row(0)
        result1 = mat.get_row(1)
        result2 = mat.get_row(2)

        self.assertTrue(np.array_equal(result0, expected0))
        self.assertTrue(np.array_equal(result1, expected1))
        self.assertTrue(np.array_equal(result2, expected2))

    def test_from_list_with_no_unique_elements(self):
        mat = SparseMatrix.from_list([[0, 0, 0, 0], [0, 0, 0, 0],
                                      [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(len(mat), 0)
        self.assertTrue(np.array_equal(mat.cols, np.array([])))
        self.assertTrue(np.array_equal(mat.data, np.array([])))
        self.assertTrue(np.array_equal(mat.rows, np.array([])))

    def test_from_list_with_several_unique_elements(self):
        mat = SparseMatrix.from_list([[0, 1, 0, 0], [0, 0, 0, 2],
                                      [4, 0, 0, 0], [0, 0, 3, 0]])
        self.assertEqual(len(mat), 4)
        self.assertTrue(np.array_equal(mat.cols, np.array([1, 3, 0, 2])))
        self.assertTrue(np.array_equal(mat.data, np.array([1, 2, 4, 3])))
        self.assertTrue(np.array_equal(mat.rows, np.array([0, 1, 2, 3])))


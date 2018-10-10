"""
Contains unit tests to ensure that the sparse matrix data structure works as
intended.
"""
import numpy as np
from unittest import TestCase

from wordcat.sparse import SparseMatrix, SparseVector


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
        mat = SparseMatrix.from_list([[0, 2, 0], [0, 0, 3], [1, 0, 0]])

        expected0 = SparseVector.from_list([0, 0, 1])
        expected1 = SparseVector.from_list([2, 0, 0])
        expected2 = SparseVector.from_list([0, 3, 0])

        result0 = mat.get_column(0)
        result1 = mat.get_column(1)
        result2 = mat.get_column(2)

        self.assertEqual(expected0, result0)
        self.assertEqual(expected1, result1)
        self.assertEqual(expected2, result2)

    def test_get_columns_using_simple_matrix(self):
        mat = SparseMatrix.from_list([[0, 2, 0], [0, 0, 3], [1, 0, 0]])

        expected = [
            SparseVector.from_list([0, 0, 1]),
            SparseVector.from_list([2, 0, 0]),
            SparseVector.from_list([0, 3, 0])
        ]
        result = mat.get_columns()

        for ex, res in zip(expected, result):
            self.assertEqual(res, ex)

    def test_get_row_using_simple_matrix(self):
        mat = SparseMatrix.from_list([[0, 2, 0], [0, 0, 3], [1, 0, 0]])

        expected0 = SparseVector.from_list([0, 2, 0])
        expected1 = SparseVector.from_list([0, 0, 3])
        expected2 = SparseVector.from_list([1, 0, 0])

        result0 = mat.get_row(0)
        result1 = mat.get_row(1)
        result2 = mat.get_row(2)

        self.assertEqual(expected0, result0)
        self.assertEqual(expected1, result1)
        self.assertEqual(expected2, result2)

    def test_get_rows_using_simple_matrix(self):
        mat = SparseMatrix.from_list([[0, 2, 0], [0, 0, 3], [1, 0, 0]])

        expected = [
            SparseVector.from_list([0, 2, 0]),
            SparseVector.from_list([0, 0, 3]),
            SparseVector.from_list([1, 0, 0])
        ]
        result = mat.get_rows()

        for ex, res in zip(expected, result):
            self.assertEqual(res, ex)

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

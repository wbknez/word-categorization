"""
Contains unit tests to verify that sparse vector operations work as intended.
"""
import numpy as np
from unittest import TestCase

from wordcat.sparse import SparseVector


class SparseVectorTest(TestCase):
    """
    Test suite for SparseVector.
    """

    def test_multiply_throws_if_vector_lengths_are_not_equal(self):
        vec0 = SparseVector.from_list([1, 2, 3, 4])
        vec1 = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4])
        vec2 = SparseVector.from_list([1, 2, 2, 3, 3, 4, 4, 5])

        with self.assertRaises(ValueError):
            _, _ = vec0.multiply(vec1)
            _, _ = vec0.multiply(vec2)
            _, _ = vec1.multiply(vec2)

    def test_multiply_with_equal_indices(self):
        vec0 = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4])
        vec1 = SparseVector.from_list([5, 0, 6, 0, 7, 0, 8])

        expected0 = SparseVector.from_list([5, 0, 12, 0, 21, 0, 32])
        expected1 = np.array([])
        result, remainder = vec0.multiply(vec1)

        self.assertEqual(result, expected0)
        self.assertTrue(np.array_equal(remainder, expected1))

    def test_multiply_with_identity(self):
        vec = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4])
        one = SparseVector.from_list([1, 1, 1, 1, 1, 1, 1])

        expected0 = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4])
        expected1 = np.array([])
        result, remainder = vec.multiply(one)

        self.assertEqual(result, expected0)
        self.assertTrue(np.array_equal(remainder, expected1))

    def test_multiply_with_unequal_indices(self):
        vec0 = SparseVector.from_list([1, 3, 2, 0, 3, 5, 4])
        vec1 = SparseVector.from_list([5, 0, 6, 4, 7, 0, 8])

        expected0 = SparseVector.from_list([5, 0, 12, 0, 21, 0, 32])
        expected1 = np.array([1, 5])
        result, remainder = vec0.multiply(vec1)

        self.assertEqual(result, expected0)
        self.assertTrue(np.array_equal(remainder, expected1))

    def test_multiply_with_unequal_indices_again(self):
        vec0 = SparseVector.from_list([0, 1, 2, 0, 3, 0, 4])
        vec1 = SparseVector.from_list([5, 3, 6, 4, 7, 5, 8])

        expected0 = SparseVector.from_list([0, 3, 12, 0, 21, 0, 32])
        expected1 = np.array([])
        result, remainder = vec0.multiply(vec1)

        self.assertEqual(result, expected0)
        self.assertTrue(np.array_equal(remainder, expected1))

    def test_multiply_with_zero(self):
        vec = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4])
        zero = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0])

        expected0 = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0])
        expected1 = np.array([0, 2, 4, 6])
        result, remainder = vec.multiply(zero)

        self.assertEqual(result, expected0)
        self.assertTrue(np.array_equal(remainder, expected1))

    def test_scale_with_random(self):
        array = np.random.randint(0, 100, 30)
        vec = SparseVector.from_list(array)
        scalar = np.random.randint(1, 32)

        expected = SparseVector.from_list(np.multiply(array, scalar))
        result = vec.scale(scalar)

        self.assertEqual(result, expected)

    def test_scale_with_zero(self):
        vec = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0])

        expected = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0])
        result = vec.scale(2)

        self.assertEqual(expected, result)

    def test_slice_with_full_vector(self):
        vec = SparseVector.from_list([1, 2, 3, 4, 5, 6, 7])

        expected = SparseVector.from_list([1, 2, 3, 4, 5, 6, 7])
        result = vec.slice([0, 1, 2, 3, 4, 5, 6])

        self.assertEqual(result, expected)

    def test_slice_with_sparse_vector(self):
        vec = SparseVector.from_list([0, 1, 0, 2, 0, 3, 0])

        expected = SparseVector.from_list([0, 0, 0, 2, 0, 3, 0])
        result = vec.slice([1, 2])

        self.assertEqual(result, expected)

    def test_slice_with_zero(self):
        vec = SparseVector.from_list([1, 2, 3, 4, 5, 6, 7])

        expected = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0])
        result = vec.slice([])

        self.assertEqual(expected, result)

    def test_sum_with_random(self):
        array = np.random.randint(0, 100, 20)
        vec = SparseVector.from_list(array)

        expected = np.sum(array)
        result = vec.sum()

        self.assertEqual(result, expected)

    def test_sum_with_zero(self):
        vec = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0])

        expected = 0
        result = vec.sum()

        self.assertEqual(result, expected)

    def test_from_list_with_no_unique_elements(self):
        vec = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.assertEqual(vec.size, 10)
        self.assertTrue(np.array_equal(vec.data, np.array([], dtype=np.uint16)))
        self.assertTrue(np.array_equal(vec.indices,
                                       np.array([], dtype=np.uint32)))

    def test_from_list_with_several_unique_elements(self):
        vec = SparseVector.from_list([0, 1, 2, 0, 3, 4, 0, 5, 6, 0, 7, 8])

        self.assertEqual(vec.size, 12)
        self.assertTrue(np.array_equal(np.array([1, 2, 3, 4, 5, 6, 7, 8],
                                                dtype=np.uint16),
                                       vec.data))
        self.assertTrue(np.array_equal(np.array([1, 2, 4, 5, 7, 8, 10, 11],
                                                dtype=np.uint32),
                                       vec.indices))

    def test_from_lists_with_no_unique_elements(self):
        vec = SparseVector.from_lists([], [], 5)

        self.assertEqual(vec.size, 5)
        self.assertTrue(np.array_equal(vec.data, np.array([], dtype=np.uint16)))
        self.assertTrue(np.array_equal(vec.indices,
                                       np.array([], dtype=np.uint32)))

    def test_from_lists_with_several_unique_elements(self):
        vec = SparseVector.from_lists([1, 2, 3, 4, 5, 6, 7, 8],
                                      [1, 2, 4, 5, 7, 8, 10, 11], 12)

        self.assertEqual(vec.size, 12)
        self.assertTrue(np.array_equal(np.array([1, 2, 3, 4, 5, 6, 7, 8],
                                                dtype=np.uint16),
                                       vec.data))
        self.assertTrue(np.array_equal(np.array([1, 2, 4, 5, 7, 8, 10, 11],
                                                dtype=np.uint32),
                                       vec.indices))

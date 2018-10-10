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

    def test_add_with_random_vector(self):
        array_a = np.array([1, 0, 2, 0, 3, 0, 4, 5])
        array_b = np.array([2, 1, 0, 3, 4, 12, 0, 7])
        a = SparseVector.from_list(array_a)
        b = SparseVector.from_list(array_b)

        expected = SparseVector.from_list([3, 0, 0, 0, 7, 0, 0, 12])
        result = a + b

        self.assertEqual(expected, result)

    def test_add_with_random_scalar(self):
        array = np.array([1, 0, 2, 0, 3, 0, 4, 5])
        vec = SparseVector.from_list(array)
        scalar = np.random.randint(1, 100)

        expected = SparseVector.from_list(array)
        expected.data = np.add(expected.data, scalar)
        expected.compact()

        result = vec + scalar

        self.assertEqual(result, expected)

    def test_add_with_zero_vector(self):
        a = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4, 5])
        b = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0, 0])

        expected = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0, 0])
        result = a + b

        self.assertEqual(expected, result)

    def test_add_with_zero_scalar(self):
        vec = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4, 5])
        scalar = 0

        expected = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4, 5])
        result = vec + scalar

        self.assertEqual(result, expected)

    def test_log2_with_random(self):
        array = np.random.randint(0, 20, 20)
        vec = SparseVector.from_list(array)

        expected = SparseVector.from_list(array)
        expected.data = np.log2(expected.data)
        result = vec.log2()

        self.assertEqual(result, expected)

    def test_log2_with_zero(self):
        array = np.random.randint(0, 1, 20)
        vec = SparseVector.from_list(array)

        expected = SparseVector.from_list(array)
        expected.data = np.log2(expected.data)
        result = vec.log2()

        self.assertEqual(result, expected)

    def test_multiply_with_random_vector(self):
        array_a = np.array([1, 0, 2, 0, 3, 0, 4, 5])
        array_b = np.random.randint(0, 100, array_a.size)
        a = SparseVector.from_list(array_a)
        b = SparseVector.from_list(array_b)

        expected = SparseVector.from_list(np.multiply(array_a, array_b))
        result = a * b

        self.assertEqual(expected, result)

    def test_multiply_with_random_scalar(self):
        array = np.array([1, 0, 2, 0, 3, 0, 4, 5])
        vec = SparseVector.from_list(array)
        scalar = np.random.randint(1, 100)

        expected = SparseVector.from_list(np.multiply(array, scalar))
        result = vec * scalar

        self.assertEqual(result, expected)

    def test_multiply_with_zero_vector(self):
        a = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4, 5])
        b = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0, 0])

        expected = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0])
        result = a * b

        self.assertEqual(expected, result)

    def test_multiply_with_zero_scalar(self):
        vec = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4, 5])
        scalar = 0

        expected = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0, 0])
        result = vec * scalar

        self.assertEqual(result, expected)

    def test_subtract_with_random_vector(self):
        array_a = np.array([1, 0, 2, 0, 3, 0, 4, 5])
        array_b = np.array([2, 1, 0, 3, 4, 12, 0, 7])
        a = SparseVector.from_list(array_a)
        b = SparseVector.from_list(array_b)

        expected = SparseVector.from_list([-1, 0, 0, 0, -1, 0, 0, -2])
        result = a - b

        self.assertEqual(expected, result)

    def test_subtract_with_random_scalar(self):
        array = np.array([1, 0, 2, 0, 3, 0, 4, 5])
        vec = SparseVector.from_list(array)
        scalar = np.random.randint(1, 100)

        expected = SparseVector.from_list(array)
        expected.data = np.subtract(expected.data, scalar)
        expected.compact()

        result = vec - scalar

        self.assertEqual(result, expected)

    def test_subtract_with_zero_vector(self):
        a = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4, 5])
        b = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0, 0])

        expected = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0, 0])
        result = a - b

        self.assertEqual(expected, result)

    def test_subtract_with_zero_scalar(self):
        vec = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4, 5])
        scalar = 0

        expected = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4, 5])
        result = vec - scalar

        self.assertEqual(result, expected)

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

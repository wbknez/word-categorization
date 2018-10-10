"""
Contains unit tests to verify that sparse vector operations work as intended.
"""
from copy import copy

import numpy as np
from unittest import TestCase

from wordcat.sparse import SparseVector


class SparseVectorTest(TestCase):
    """
    Test suite for SparseVector.
    """

    def test_abs_with_random(self):
        array = np.random.randint(-20, 20, 20)
        vec = SparseVector.from_list(array)

        expected = SparseVector.from_list(array)
        expected.data = np.abs(expected.data)
        result = vec.abs()

        self.assertEqual(result, expected)

    def test_abs_with_zero(self):
        array = np.random.randint(0, 1, 20)
        vec = SparseVector.from_list(array)

        expected = SparseVector.from_list(array)
        result = vec.abs()

        self.assertEqual(result, expected)

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
        expected.data = np.add(expected.data,
                               np.full(expected.data.size, scalar))
        expected.compact()

        result = vec + scalar

        self.assertEqual(result, expected)

    def test_add_with_zero_vector(self):
        a = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4, 5])
        b = SparseVector.zero(8)

        expected = SparseVector.zero(8)
        result = a + b

        self.assertEqual(expected, result)

    def test_add_with_zero_scalar(self):
        vec = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4, 5])
        scalar = 0

        expected = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4, 5])
        result = vec + scalar

        self.assertEqual(result, expected)

    def test_compact_with_random(self):
        data = np.random.randint(0, 10, 30, dtype=np.uint16)
        indices = np.arange(30, dtype=np.uint32)

        vec = SparseVector(data, indices, 30)
        vec.compact()

        zi = np.where(data == 0)
        expected = SparseVector(np.delete(data, zi), np.delete(indices, zi), 30)
        result = copy(vec)

        self.assertEqual(result, expected)

    def test_compact_with_zero(self):
        vec = SparseVector(
            np.zeros(10, dtype=np.uint16), np.arange(10, dtype=np.uint32), 10
        )
        vec.compact()

        expected = SparseVector.zero(10)
        result = copy(vec)

        self.assertEqual(result, expected)

    def test_divide_with_random_vector(self):
        array_a = np.random.randint(0, 100, 30)
        array_b = np.random.randint(0, 100, array_a.size)
        a = SparseVector.from_list(array_a)
        b = SparseVector.from_list(array_b)

        a_idx = np.in1d(a.indices, b.indices)
        b_idx = np.in1d(b.indices, a.indices)

        expected = SparseVector.from_list(array_a)
        expected.data = np.divide(a.data[a_idx], b.data[b_idx])
        expected.indices = a.indices[a_idx]
        expected.size = a.size
        result = a / b

        self.assertEqual(result, expected)

    def test_divide_with_random_scalar(self):
        array = np.array([1, 0, 2, 0, 3, 0, 4, 5])
        vec = SparseVector.from_list(array)
        scalar = np.random.randint(1, 100)

        expected = SparseVector.from_list(array)
        expected.data = np.divide(expected.data, scalar)
        result = vec / scalar

        self.assertEqual(result, expected)

    def test_divide_with_zero_vector(self):
        a = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4, 5])
        b = SparseVector.zero(8)

        expected = SparseVector.zero(8)
        result = a / b

        self.assertEqual(expected, result)

    def test_divide_with_zero_scalar_throws(self):
        with self.assertRaises(ZeroDivisionError):
            _ = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4, 5]) / 0

    def test_exp_with_random(self):
        array = np.random.randint(0, 20, 20)
        vec = SparseVector.from_list(array)

        expected = SparseVector.from_list(array)
        expected.data = np.exp(expected.data)
        result = vec.exp()

        self.assertEqual(result, expected)

    def test_exp_with_zero(self):
        array = np.random.randint(0, 1, 20)
        vec = SparseVector.from_list(array)

        expected = SparseVector.from_list(array)
        expected.data = np.exp(expected.data)
        result = vec.exp()

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
        b = SparseVector.zero(8)

        expected = SparseVector.zero(8)
        result = a * b

        self.assertEqual(expected, result)

    def test_multiply_with_zero_scalar(self):
        vec = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4, 5])
        scalar = 0

        expected = SparseVector.zero(8)
        result = vec * scalar

        self.assertEqual(result, expected)

    def test_negate_with_random(self):
        array = np.random.randint(0, 100, 20)
        vec = SparseVector.from_list(array)

        expected = SparseVector.from_list(array)
        expected.data = np.negative(expected.data)
        result = -vec

        self.assertEqual(result, expected)

    def test_negate_with_zero(self):
        vec = SparseVector.zero(4)

        expected = SparseVector.zero(4)
        result = -vec

        self.assertEqual(result, expected)

    def test_power_with_random(self):
        array = np.random.randint(0, 20, 20)
        vec = SparseVector.from_list(array)
        a = np.random.randint(2, 10)

        expected = SparseVector.from_list(array)
        expected.data = np.power(expected.data, a)
        result = vec.power(a)

        self.assertEqual(result, expected)

    def test_power_with_zero(self):
        array = np.random.randint(0, 1, 20)
        vec = SparseVector.from_list(array)
        a = np.random.randint(2, 10)

        expected = SparseVector.from_list(array)
        expected.data = np.power(expected.data, a)
        result = vec.power(a)

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
        expected.data = np.subtract(expected.data,
                                    np.full(expected.data.size, scalar))
        expected.compact()

        result = vec - scalar

        self.assertEqual(result, expected)

    def test_subtract_with_zero_vector(self):
        a = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4, 5])
        b = SparseVector.zero(8)

        expected = SparseVector.zero(8)
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
        vec = SparseVector.zero(7)

        expected = 0
        result = vec.sum()

        self.assertEqual(result, expected)

    def test_venn_with_random(self):
        array_a = np.random.randint(0, 100, 30)
        array_b = np.random.randint(0, 100, 30)

        a = SparseVector.from_list(array_a)
        b = SparseVector.from_list(array_b)

        a_i = np.in1d(a.indices, b.indices)
        a_d = np.in1d(a.indices,
                      np.setdiff1d(a.indices, a.indices[a_i]))

        expected0 = SparseVector(a.data[a_i], a.indices[a_i], a.size)
        expected1 = SparseVector(a.data[a_d], a.indices[a_d], a.size)

        result0, result1 = a.venn(b)

        self.assertEqual(result0, expected0)
        self.assertEqual(result1, expected1)
        self.assertEqual(result0.data.size + result1.data.size, a.data.size)
        self.assertEqual(result0.indices.size + result1.indices.size,
                         a.indices.size)

    def test_venn_with_zero(self):
        a = SparseVector.from_list([1, 0, 2, 0, 3, 4, 0, 5])
        b = SparseVector.zero(8)

        expected0 = SparseVector.zero(8)
        expected1 = SparseVector.from_list([1, 0, 2, 0, 3, 4, 0, 5])

        result0, result1 = a.venn(b)

        self.assertEqual(result0, expected0)
        self.assertEqual(result1, expected1)

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

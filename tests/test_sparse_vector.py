"""

"""
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

        expected = SparseVector.from_list([5, 0, 12, 0, 21, 0, 32])
        result, remainder = vec0.multiply(vec1)

        self.assertEqual(result, expected)
        self.assertEqual(remainder, 0)

    def test_multiply_with_identity(self):
        vec = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4])
        one = SparseVector.from_list([1, 1, 1, 1, 1, 1, 1])

        expected = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4])
        result, remainder = vec.multiply(one)

        self.assertEqual(result, expected)
        self.assertEqual(remainder, 0)

    def test_multiply_with_zero(self):
        vec = SparseVector.from_list([1, 0, 2, 0, 3, 0, 4])
        zero = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0])

        expected = SparseVector.from_list([0, 0, 0, 0, 0, 0, 0])
        result, remainder = vec.multiply(zero)

        self.assertEqual(result, expected)
        self.assertEqual(remainder, 4)

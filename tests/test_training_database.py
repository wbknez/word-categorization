"""
Contains unit tests to verify that the sparse database of word counts behaves
as expected.
"""
from copy import copy

import numpy as np
from unittest import TestCase

from wordcat.sparse import SparseMatrix, SparseVector
from wordcat.storage import TrainingDatabase


class DatabaseTest(TestCase):
    """
    Test suite for TrainingDatabase.
    """

    def test_class_frequencies_with_single_class(self):
        tdb = TrainingDatabase(np.full(1200, 3, dtype=np.uint8),
                               SparseMatrix.zero((1200, 100)))

        expected = {3: 1.0}
        result = tdb.create_class_frequency_table()

        self.assertEqual(result, expected)

    def test_class_frequencies_with_multiple_classes(self):
        tdb = TrainingDatabase(np.random.randint(1, 8, 1200, dtype=np.uint8),
                               SparseMatrix.zero((1200, 100)))
        class_counts, frequencies = np.unique(tdb.classes, return_counts=True)

        expected = {
            cls: (freq / 1200) for cls, freq in zip(class_counts, frequencies)
        }
        result = tdb.create_class_frequency_table()

        self.assertTrue(result, expected)

    def test_select_with_multiple_classes(self):
        classes = np.array([1, 1, 2, 2, 3, 3, 2, 1, 3, 1], dtype=np.uint8)
        vectors = [SparseVector.random(0, 5, 10) for _ in range(10)]

        tdb = TrainingDatabase(classes, SparseMatrix.vstack(vectors))

        expected = [
            SparseMatrix.vstack([vectors[i] for i in [0, 1, 7, 9]]),
            SparseMatrix.vstack([vectors[i] for i in [2, 3, 6]]),
            SparseMatrix.vstack([vectors[i] for i in [4, 5, 8]])
        ]
        result = [tdb.select(i) for i in np.unique(classes)]

        for exp, res in zip(expected, result):
            self.assertEqual(res, exp)

    def test_select_with_single_class(self):
        tdb = TrainingDatabase(np.full(10, 1, dtype=np.uint8),
                               SparseMatrix.random(0, 5, (10, 3)))

        expected = copy(tdb.counts)
        result = tdb.select(1)

        self.assertEqual(result, expected)

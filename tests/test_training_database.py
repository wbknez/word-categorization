"""
Contains unit tests to verify that the sparse database of word counts behaves
as expected.
"""
import numpy as np
from unittest import TestCase

from wordcat.sparse import SparseMatrix
from wordcat.storage import TrainingDatabase


class DatabaseTest(TestCase):
    """
    Test suite for TrainingDatabase.
    """

    def test_class_frequencies_with_single_class(self):
        tdb = TrainingDatabase(np.full(1200, 3, dtype=np.uint8),
                               SparseMatrix(np.array([], dtype=np.uint16),
                                            np.array([], dtype=np.uint32),
                                            np.array([], dtype=np.uint32),
                                            (1200, 100)))
        clszs, freqs = tdb.get_class_frequencies()

        self.assertEqual(clszs.size, 1)
        self.assertEqual(freqs.size, 1)
        self.assertEqual(clszs[0], 3)
        self.assertEqual(freqs[0], 1.0)

    def test_class_frequencies_with_multiple_classes(self):
        tdb = TrainingDatabase(np.random.randint(1, 8, 1200, dtype=np.uint8),
                               SparseMatrix(np.array([], dtype=np.uint16),
                                            np.array([], dtype=np.uint32),
                                            np.array([], dtype=np.uint32),
                                            (1200, 100)))
        clszs, freqs = tdb.get_class_frequencies()
        uniques, counts = np.unique(tdb.classes, return_counts=True)

        self.assertEqual(clszs.size, uniques.size)
        self.assertEqual(freqs.size, counts.size)
        self.assertTrue(np.array_equal(clszs, uniques))
        self.assertTrue(np.array_equal(freqs,
                                       np.divide(counts, tdb.classes.size)))

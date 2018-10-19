"""
Contains unit tests to verify that the sparse database of word counts behaves
as expected.
"""
from copy import copy

import numpy as np
from unittest import TestCase

from wordcat.sparse import SparseMatrix, SparseVector
from wordcat.storage import TrainingDatabase, TestingSet, Test


class DatabaseTest(TestCase):
    """
    Test suite for TrainingDatabase.
    """

    def test_create_deltas_with_single_class(self):
        tdb = TrainingDatabase(np.full(1200, 3, dtype=np.uint8),
                               SparseMatrix.zero((1200, 100)))

        expected = np.zeros((4, tdb.counts.row_count), dtype=np.uint8)
        expected[3, :] = 1
        result = tdb.create_deltas()

        self.assertTrue(np.array_equal(result, expected))

    def test_create_deltas_with_multiple_classes(self):
        tdb = TrainingDatabase(np.random.randint(1, 4, 1200, dtype=np.uint8),
                               SparseMatrix.zero((1200, 100)))

        expected = np.zeros((4, tdb.counts.row_count), dtype=np.uint8)

        for i in range(4):
            expected[i, np.where(tdb.classes == i)] = 1

        result = tdb.create_deltas()

        self.assertTrue(np.array_equal(result, expected))

    def test_create_frequencies_with_single_class(self):
        tdb = TrainingDatabase(np.full(1200, 3, dtype=np.uint8),
                               SparseMatrix.zero((1200, 100)))

        expected = {3: 1.0}
        result = tdb.create_frequencies()

        self.assertEqual(result, expected)

    def test_create_frequencies_with_multiple_classes(self):
        tdb = TrainingDatabase(np.random.randint(1, 8, 1200, dtype=np.uint8),
                               SparseMatrix.zero((1200, 100)))
        class_counts, frequencies = np.unique(tdb.classes, return_counts=True)

        expected = {
            cls: (freq / 1200) for cls, freq in zip(class_counts, frequencies)
        }
        result = tdb.create_frequencies()

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

    def test_shuffle(self):
        tdb = TrainingDatabase(np.array([0, 1, 2, 3], dtype=np.uint8),
                               SparseMatrix.from_list([
                                   [1, 2, 3, 4, 5],
                                   [6, 7, 8, 9, 10],
                                   [11, 12, 13, 14, 15],
                                   [16, 17, 18, 19, 20]
                               ]))
        mirror = copy(tdb)

        for _ in range(100):
            tdb.shuffle()

        expected0 = np.unique(mirror.classes, return_counts=True)
        result0 = np.unique(tdb.classes, return_counts=True)

        self.assertTrue(result0, expected0)

        counts = 0
        row_set = set([row for row in tdb.counts.get_rows()])


        for row in mirror.counts.get_rows():
            if row in row_set:
                counts += 1

        self.assertEqual(counts, mirror.counts.row_count)

    def test_split_with_simple_matrix(self):
        tdb = TrainingDatabase(np.array([0, 1, 1, 0], dtype=np.uint8),
                               SparseMatrix.from_list([
                                   [0, 0, 0, 0],
                                   [2, 2, 2, 2],
                                   [3, 3, 3, 3],
                                   [1, 1, 1, 1]
                               ]))

        expected0 = TrainingDatabase(np.array([0, 0], dtype=np.uint8),
                                     SparseMatrix.from_list([
                                         [0, 0, 0, 0],
                                         [1, 1, 1, 1]
                                     ]))
        expected1 = TestingSet([
            Test(0, SparseVector.from_list([2, 2, 2, 2])),
            Test(1, SparseVector.from_list([3, 3, 3, 3]))
        ])
        expected2 = np.array([1, 1], dtype=np.uint8)

        result0, result1, result2 = tdb.split(1, 3)

        self.assertEqual(result0, expected0)
        self.assertEqual(result1, expected1)
        self.assertTrue(np.array_equal(result2, expected2))

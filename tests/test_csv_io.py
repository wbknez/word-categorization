"""
Contains unit tests to ensure that parallel CSV I/O operations work as expected.
"""
from io import StringIO
from multiprocessing import Pool

import numpy as np
from unittest import TestCase

from wordcat.io import CsvIO
from wordcat.sparse import SparseMatrix, SparseVector
from wordcat.storage import TestingSet, TrainingDatabase


class CsvIOTest(TestCase):
    """
    Test suite for CsvIO.
    """

    def test_read_database_with_multiple_rows(self):
        src = "0,0,1,0,2,0,3,4,0,5,1\n" +\
              "1,6,0,7,0,8,9,0,10,0,2\n" +\
              "2,0,11,0,12,0,13,14,0,15,1\n"

        with Pool(processes=4) as pool, StringIO(src) as stream:
            expected = TrainingDatabase(np.array([1, 2, 1], dtype=np.uint8),
                                        SparseMatrix.from_list([
                                            [0, 1, 0, 2, 0, 3, 4, 0, 5],
                                            [6, 0, 7, 0, 8, 9, 0, 10, 0],
                                            [0, 11, 0, 12, 0, 13, 14, 0, 15]
                                        ]))
            expected.counts.cols = expected.counts.cols + 1
            expected.counts.shape = (expected.counts.shape[0],
                                     expected.counts.shape[1] + 1)

            result = CsvIO.read_database(pool, stream)

            self.assertEqual(result, expected)

    def test_read_database_with_single_row(self):
        src = "0,0,1,0,2,0,3,4,0,5,1"

        with Pool(processes=4) as pool, StringIO(src) as stream:
            expected = TrainingDatabase(np.ones(1, dtype=np.uint8),
                                        SparseMatrix.from_list([
                                            [0, 1, 0, 2, 0, 3, 4, 0, 5],
                                        ]))
            expected.counts.cols = expected.counts.cols + 1
            expected.counts.shape = (expected.counts.shape[0],
                                     expected.counts.shape[1] + 1)

            result = CsvIO.read_database(pool, stream)

            self.assertEqual(result, expected)

    def test_read_set_with_multiple_rows(self):
        src = "0,0,1,0,2,0,3,4,0,5\n" + \
              "1,6,0,7,0,8,9,0,10,0\n" + \
              "2,0,11,0,12,0,13,14,0,15\n"

        with Pool(processes=4) as pool, StringIO(src) as stream:
            expected = TestingSet(
                [0, 1, 2], [
                    SparseVector.from_list([0, 1, 0, 2, 0, 3, 4, 0, 5]),
                    SparseVector.from_list([6, 0, 7, 0, 8, 9, 0, 10, 0]),
                    SparseVector.from_list([0, 11, 0, 12, 0, 13, 14, 0, 15])
                ]
            )

            for i in range(3):
                expected.tests[i].indices = expected.tests[i].indices + 1
                expected.tests[i].size = expected.tests[i].size + 1

            result = CsvIO.read_set(pool, stream)

            self.assertEqual(result, expected)

    def test_read_set_with_single_row(self):
        src = "0,0,1,0,2,0,3,4,0,5"

        with Pool(processes=4) as pool, StringIO(src) as stream:
            expected = TestingSet(
                [0], [SparseVector.from_list([0, 1, 0, 2, 0, 3, 4, 0, 5])]
            )
            expected.tests[0].indices = expected.tests[0].indices + 1
            expected.tests[0].size = expected.tests[0].size + 1

            result = CsvIO.read_set(pool, stream)

            self.assertEqual(result, expected)

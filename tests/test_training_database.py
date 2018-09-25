"""
Contains unit tests to verify that the sparse database of word counts behaves
as expected.
"""
import numpy as np
from unittest import TestCase

from scipy import sparse

from wordcat.storage import TrainingDatabase


class DatabaseTest(TestCase):
    """
    Test suite for TrainingDatabase.
    """

    def test_init_throws_when_number_of_classes_is_wrong(self):
        with self.assertRaises(ValueError):
            TrainingDatabase(np.zeros(10, np.int16),
                             sparse.csr_matrix(np.zeros((9, 10), np.int)))
            TrainingDatabase(np.zeros(10, np.int16),
                             sparse.csr_matrix(np.zeros((11, 10), np.int)))
            TrainingDatabase(np.zeros(10, np.int16),
                             sparse.csr_matrix(np.zeros((10, 9), np.int)))
            TrainingDatabase(np.zeros(10, np.int16),
                             sparse.csr_matrix(np.zeros((10, 11), np.int)))

    def test_init_throws_when_matrix_type_is_wrong(self):
        with self.assertRaises(ValueError):
            TrainingDatabase(np.zeros(10, np.int16),
                             sparse.csc_matrix(np.zeros((10, 10), np.int)))

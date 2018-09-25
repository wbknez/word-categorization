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

    def test_init_throws_when_type_of_classes_is_wrong(self):
        with self.assertRaises(ValueError):
            TrainingDatabase(np.zeros(10, np.float),
                             sparse.csr_matrix(np.zeros((10, 10), np.uint16)))
            TrainingDatabase(np.zeros(10, np.int),
                             sparse.csr_matrix(np.zeros((10, 10), np.uint16)))
            TrainingDatabase(np.zeros(10, np.int8),
                             sparse.csr_matrix(np.zeros((10, 10), np.uint16)))
            TrainingDatabase(np.zeros(10, np.int16),
                             sparse.csr_matrix(np.zeros((10, 10), np.uint16)))
            TrainingDatabase(np.zeros(10, np.int32),
                             sparse.csr_matrix(np.zeros((10, 10), np.uint16)))
            TrainingDatabase(np.zeros(10, np.uint),
                             sparse.csr_matrix(np.zeros((10, 10), np.uint16)))
            TrainingDatabase(np.zeros(10, np.uint16),
                             sparse.csr_matrix(np.zeros((10, 10), np.uint16)))
            TrainingDatabase(np.zeros(10, np.uint32),
                             sparse.csr_matrix(np.zeros((10, 10), np.uint16)))

    def test_init_throws_when_type_of_counts_is_wrong(self):
        with self.assertRaises(ValueError):
            TrainingDatabase(np.zeros(10, np.uint8),
                             sparse.csr_matrix(np.zeros((10, 10), np.float)))
            TrainingDatabase(np.zeros(10, np.uint8),
                             sparse.csr_matrix(np.zeros((10, 10), np.int)))
            TrainingDatabase(np.zeros(10, np.uint8),
                             sparse.csr_matrix(np.zeros((10, 10), np.int8)))
            TrainingDatabase(np.zeros(10, np.uint8),
                             sparse.csr_matrix(np.zeros((10, 10), np.int16)))
            TrainingDatabase(np.zeros(10, np.uint8),
                             sparse.csr_matrix(np.zeros((10, 10), np.int32)))
            TrainingDatabase(np.zeros(10, np.uint8),
                             sparse.csr_matrix(np.zeros((10, 10), np.uint)))
            TrainingDatabase(np.zeros(10, np.uint8),
                             sparse.csr_matrix(np.zeros((10, 10), np.uint8)))
            TrainingDatabase(np.zeros(10, np.uint8),
                             sparse.csr_matrix(np.zeros((10, 10), np.uint32)))

    def test_init_throws_when_number_of_classes_is_wrong(self):
        with self.assertRaises(ValueError):
            TrainingDatabase(np.zeros(10, np.uint8),
                             sparse.csr_matrix(np.zeros((9, 10), np.uint16)))
            TrainingDatabase(np.zeros(10, np.uint8),
                             sparse.csr_matrix(np.zeros((11, 10), np.uint16)))
            TrainingDatabase(np.zeros(10, np.uint8),
                             sparse.csr_matrix(np.zeros((10, 9), np.uint16)))
            TrainingDatabase(np.zeros(10, np.uint8),
                             sparse.csr_matrix(np.zeros((10, 11), np.uint16)))

    def test_init_throws_when_sparse_matrix_type_is_wrong(self):
        with self.assertRaises(ValueError):
            TrainingDatabase(np.zeros(10, np.uint8),
                             sparse.csc_matrix(np.zeros((10, 10), np.uint16)))

"""
Contains all classes and functions related to storing data for a learning
algorithm to use and operate upon.
"""
import numpy as np

from wordcat.utils import sparse_equal


class TrainingDatabase:
    """
    Represents a table of data that a learning algorithm may use to train
    itself and thereby learn about the hypothesis space in which it is
    expected to operate.

    Please note that this object is sparse and, since it is intended to keep
    memory low, has some severe restrictions in terms of format.  In
    particular:
      - All classes are expected to be 8-bit unsigned integers.
      - All counts are expected to be 16-bit unsigned integers.
    
    Attributes:
        classes (np.array): The collection of results.
        counts (sparse.csr_matrix): The collection of word counts.
    """

    def __init__(self, classes, counts):
        if len(classes.shape) != 1:
            raise ValueError("Classes must be one dimensional")
        if len(counts.shape) != 2:
            raise ValueError("Counts must be two dimensional.")
        if classes.shape[0] != counts.shape[0]:
            raise ValueError("The number of classes must equal the number of "
                             "rows.")
        if not counts.getformat() == "csr":
            raise ValueError("The matrix type must be CSR.")

        if not classes.dtype == np.uint8:
            raise ValueError("Classes must be 8-bit unsigned integers.")
        if not counts.dtype == np.uint16:
            raise ValueError("Counts must be 16-bit unsigned integers.")

        self.classes = classes
        self.counts = counts

    def __eq__(self, other):
        if isinstance(other, TrainingDatabase):
            return np.array_equal(self.classes, other.classes) and \
                   sparse_equal(self.counts, other.counts)
        return NotImplemented

    def __getattr__(self, item):
        if item == "cols":
            return self.counts.shape[1]
        elif item == "rows":
            return self.counts.shape[0]

    def __getitem__(self, item):
        row, col = item
        return self.counts[row, col]

    def __ne__(self, other):
        return not self == other

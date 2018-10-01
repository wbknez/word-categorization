"""
Contains all classes and functions related to storing data for a learning
algorithm to use and operate upon.
"""
import numpy as np


class TestingSet:
    """
    Represents a collection of test data.

    Attributes:
        ids (list): The list of unique ids, one per test.
        tests (list): The list of tests, where each test is a sparse vector
        of word counts.
    """

    def __init__(self, ids, tests):
        if len(ids) != len(tests):
            raise ValueError("The number of unique ids must equal the number "
                             "of tests: {} not {}.".format(len(ids),
                                                           len(tests)))

        self.ids = ids
        self.tests = tests

    def __eq__(self, other):
        if isinstance(other, TestingSet):
            return self.ids == other.ids and self.tests == other.tests
        return NotImplemented

    def __iter__(self):
        for id, test in zip(self.ids, self.tests):
            yield id, test

    def __ne__(self, other):
        return not self == other


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
      - All count indices are expected, but not required, to be 32-bit
      unsigned integers.
    
    Attributes:
        classes (np.array):
        counts (SparseMatrix):
    """

    def __init__(self, classes, counts):
        if classes.dtype != np.uint8:
            raise ValueError("Classes are expected to be 8-bit unsigned "
                             "integers.")
        if counts.dtype != np.uint16:
            raise ValueError("Counts are expected to be 16-bit unsigned "
                             "integers.")
        if counts.shape[0] != classes.size:
            raise ValueError("The number of rows does not match the number of "
                             "classes.  The number of rows is: {} but the "
                             "number of classes is: "
                             "{}".format(counts.shape[0], classes.size))

        self.classes = classes
        self.counts = counts

    def __getattr__(self, item):
        if item == "cols":
            return self.counts.shape[1]
        elif item == "data":
            return self.counts.data
        elif item == "dtype":
            return self.counts.dtype
        elif item == "rows":
            return self.counts.shape[0]
        elif item == "shape":
            return self.counts.shape

    def get_class_frequencies(self):
        """
        Computes and returns the frequencies of all unique classes in this
        training database.

        The returned value is actually a tuple of two items:
          1. An array of all unique classes found in this training database, and
          2. an array of their computed frequencies relative to one another.

        The formula for a single frequency is given as:
            f(C_i) = (number of classes of type C_i) / (total number of classes)

        Please note that this function does not ensure a full list.  Put
        another way, any classes that are not found (because there are no
        examples for them) are not included in the frequency computation and
        are also not included in the returned class (id) list.

        :return: A tuple consisting of one array of unique classes and
        another of their frequencies.
        """
        class_counts = np.unique(self.classes, return_counts=True)
        return class_counts[0], np.divide(class_counts[1], len(self.classes))

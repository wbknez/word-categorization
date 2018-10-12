"""
Contains all classes and functions related to storing data for a learning
algorithm to use and operate upon.
"""
from collections import namedtuple

import numpy as np

from wordcat.sparse import SparseMatrix


class ClassLabels:
    """
    Represents a collection of textual labels for an arbitrary number of
    classes.

    Attributes:
        classes (dict): The list of class labels.
    """

    def __init__(self, classes):
        self.classes = classes

    def __eq__(self, other):
        if isinstance(other, ClassLabels):
            return self.classes == other.classes
        return NotImplemented

    def __getattr__(self, item):
        if item == "count":
            return len(self.classes)

    def __getitem__(self, item):
        return self.classes[item]

    def __getstate__(self):
        return self.__dict__.copy()

    def __iter__(self):
        for id, classz in self.classes.items():
            yield id, classz

    def __ne__(self, other):
        return not self == other

    def __setstate__(self, state):
        self.__dict__.update(state)


class Test(namedtuple("Test", ["id", "query"])):
    """
    Represents a single test query with associated id with which a learning
    algorithm may use to make a prediction.
    """

    pass


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

        self.tests = [Test(id, test) for id, test in zip(ids, tests)]

    def __eq__(self, other):
        if isinstance(other, TestingSet):
            return self.tests == other.tests
        return NotImplemented

    def __getitem__(self, item):
        return self.tests[item]

    def __iter__(self):
        for test in self.tests:
            yield test

    def __hash__(self):
        return hash(self.tests)

    def __len__(self):
        return len(self.tests)

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

    def __eq__(self, other):
        if isinstance(other, TrainingDatabase):
            return np.array_equal(self.classes, other.classes) and \
                   self.counts == other.counts
        return NotImplemented

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

    def __getstate__(self):
        return self.__dict__.copy()

    def __ne__(self, other):
        return not self == other

    def __setstate__(self, state):
        self.__dict__.update(state)

    def create_class_frequency_table(self):
        """
        Computes and returns the frequencies of all unique classes in this
        training database associated by identifier.

        The returned value is actually a tuple of two items:
          1. An array of all unique classes found in this training database, and
          2. an array of their computed frequencies relative to one another.

        The formula for a single frequency is given as:
            f(C_i) = (number of classes of type C_i) / (total number of classes)

        Please note that this function does not ensure a full list.  Put
        another way, any classes that are not found (because there are no
        examples for them) are not included in the frequency computation and
        are also not included in the returned class (id) mapping.

        :return: A mapping of class frequencies by identifier.
        """
        class_counts, frequencies = np.unique(self.classes, return_counts=True)
        frequencies = np.divide(frequencies, len(self.classes))

        return {cls: freq for cls, freq in zip(class_counts, frequencies)}

    def select(self, classz):
        """
        Computes the sub-matrix that represents all training examples whose
        classification is the specified class.

        :param classz: The class to select.
        :return: A sparse matrix of data specific to a class.
        """
        indices = np.where(self.classes == classz)[0]
        return SparseMatrix.vstack([self.counts.get_row(i) for i in indices])

    def shuffle(self):
        """

        """
        items = [
            self.classes,
            self.counts.cols,
            self.counts.data,
            self.counts.rows,
        ]
        state = np.random.get_state()

        for item in items:
            np.random.set_state(state)
            np.random.shuffle(item)


class Vocabulary:
    """
    Represents a collection of words found across a large span of documents.

    Please note that for this project, the list of words is offset by one due to
    other data's stipulation that words are indexed starting from one.  The
    count operation on this class returns the true number of words and not the
    length of the offset list.

    Attributes:
        words (list): The list of words, off-set by one.
    """

    def __init__(self, words):
        self.words = words

        if self.words[0]:
            self.words.insert(0, "")

    def __eq__(self, other):
        if isinstance(other, Vocabulary):
            return self.words == other.words
        return NotImplemented

    def __getattr__(self, item):
        if item == "count":
            return len(self.words) - 1

    def __getitem__(self, item):
        return self.words[item]

    def __getstate__(self):
        return self.__dict__.copy()

    def __iter__(self):
        for index, word in enumerate(self.words, 1):
            yield index, word

    def __ne__(self, other):
        return not self == other

    def __setstate__(self, state):
        self.__dict__.update(state)

"""
Contains all classes and functions related to storing data for a learning
algorithm to use and operate upon.
"""
from collections import namedtuple
from copy import copy

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


class ConfusionMatrix:
    """
    Represents a matrix of predictive results used to determine and visualize
    any correlations between which classes a learning algorithm "confuses",
    or misclassifies, with another.

    Attributes:
        counts (dict): The collection of classification counts associated
        with id.
    """

    def __init__(self, labels):
        self.counts = {}

        for id, _ in labels:
            self.counts[id] = {oid: 0 for oid, _ in labels}

    def update(self, validated_predictions):
        """
        Updates this confusion matrix's class count collection with the
        specified validated results.

        :param validated_predictions: The validated predictions to use.
        """
        for valpred in validated_predictions:
            self.counts[valpred.answer][valpred.prediction.result] += 1


class Fold(namedtuple("Fold", ["start", "end"])):
    """
    Represents a single slice of a training database that should be used
    instead for testing and validation.
    """

    pass


class Prediction(namedtuple("Prediction", ["id", "result"])):
    """
    Represents a single predictive result generated by querying a learning
    algorithm with test input.
    """

    pass


class Test(namedtuple("Test", ["id", "query"])):
    """
    Represents a single test query and associated id with which a learning
    algorithm may use to make a prediction.
    """

    pass


class TestingSet:
    """
    Represents a collection of test data.

    Attributes:
        tests (list): The list of tests, where each test is a sparse vector
        of word counts with an associated id.
    """

    def __init__(self, tests):
        self.tests = tests

    def __eq__(self, other):
        if isinstance(other, TestingSet):
            return self.tests == other.tests
        return NotImplemented

    def __getitem__(self, item):
        return self.tests[item]

    def __hash__(self):
        return hash(self.tests)

    def __iter__(self):
        for test in self.tests:
            yield test

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

    def __copy__(self):
        return TrainingDatabase(np.copy(self.classes), copy(self.counts))

    def __eq__(self, other):
        if isinstance(other, TrainingDatabase):
            return np.array_equal(self.classes, other.classes) and \
                   self.counts == other.counts
        return NotImplemented

    def __getattr__(self, item):
        if item == "col_count":
            return self.counts.shape[1]
        elif item == "data":
            return self.counts.data
        elif item == "dtype":
            return self.counts.dtype
        elif item == "row_count":
            return self.counts.shape[0]
        elif item == "shape":
            return self.counts.shape

    def __getstate__(self):
        return self.__dict__.copy()

    def __ne__(self, other):
        return not self == other

    def __setstate__(self, state):
        self.__dict__.update(state)

    def create_deltas(self):
        """
        Computes and returns the matrix of binary classifications per unique
        class in this training database; that is, each row represents which
        examples belong to a class (given by a one) or not (denoted as zero).

        :return: A matrix of example classifications by class.
        """
        num_classes = np.max(self.classes) + 1
        deltas = np.zeros((num_classes, self.counts.row_count), dtype=np.uint8)

        for classz in range(num_classes):
            indices = np.where(self.classes == classz)
            deltas[classz, indices] = 1

        return deltas

    def create_k_folds(self, k):
        """
        Creates k "folds", or indexed divisions, that may be used to
        subdivide this database into a new training database plus validation
        set.

        :param k: The number of indexed folds to create.
        :return: A collection of indices representing folds.
        """
        if k < 1:
            raise ValueError("K must be at least one or greater.")

        ratio = np.floor(self.counts.row_count / k)
        folds = [ratio * i for i in range(k + 1)]
        folds[-1] = self.counts.row_count

        return [Fold(folds[j], folds[j + 1]) for j in range(k)]

    def create_frequencies(self):
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
        Shuffles this training database, re-arranging the order of both the
        classes and data by row only.
        """
        rows = [row for row in self.counts.get_rows()]
        state = np.random.get_state()

        for item in [self.classes, rows]:
            np.random.set_state(state)
            np.random.shuffle(item)

        self.counts = SparseMatrix.vstack(rows, dtype=self.counts.data.dtype)

    def split(self, start, end):
        """
        Splits this training database into two items: a new database formed
        from removing the elements from the specified interval and a testing
        set created from the rows taken.

        :param start: The starting row, inclusive.
        :param end: The ending row, exclusive.
        :return: A tuple consisting of a new database free of a selection of
        rows, a testing set made from the remainder, and the true classes to
        validate the set.
        """
        if end > self.counts.row_count:
            raise ValueError("End must be less than the number of rows.")
        if start < 0:
            raise ValueError("Start must be positive.")
        if start >= self.counts.row_count:
            raise ValueError("Start must be less than the number of rows.")
        if start >= end:
            raise ValueError("Start must be smaller than end.")

        difference = end - start
        modified = np.where(self.counts.rows >= end)[0]
        removed = np.where((self.counts.rows >= start) &
                           (self.counts.rows < end))[0]

        ntdb = copy(self)
        ts = TestingSet(
            [(i - start, ntdb.counts.get_row(i)) for i in range(start, end)]
        )
        classes = ntdb.classes[slice(start, end)]

        ntdb.counts.rows[modified] = ntdb.counts.rows[modified] - difference

        ntdb.classes = np.delete(ntdb.classes, slice(start, end))
        ntdb.counts.cols = np.delete(ntdb.counts.cols, removed)
        ntdb.counts.data = np.delete(ntdb.counts.data, removed)
        ntdb.counts.rows = np.delete(ntdb.counts.rows, removed)
        ntdb.counts.shape = (ntdb.counts.row_count - difference,
                             ntdb.counts.col_count)

        return ntdb, ts, classes


class ValidatedPrediction(namedtuple("ValidatedPrediction",
                                     ["answer", "prediction"])):
    """
    Represents a single predictive result from a learning algorithm that is
    matched with its validated, correct answer.
    """

    def is_correct(self):
        """
        Returns whether or not this validated result was predicted correctly
        or not.

        :return: Whether or not a prediction was correct.
        """
        return self.answer == self.prediction.result


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

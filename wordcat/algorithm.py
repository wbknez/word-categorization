"""
Contains all classes and functions related to implementing the ore portion of
this project - the actual machine learning algorithms themselves.
"""
from abc import ABCMeta, abstractmethod


class LearningAlgorithm(metaclass=ABCMeta):
    """
    Represents a machine learning algorithm that attempts to use
    probabilistic methods to learn patterns in training data and thereby make
    accurate predictions for queries relating to that data.

    For this project, learning algorithms explore topic categorization
    through the use of vocabulary data; that is, matching the topic category
    a word or phrase is associated with.

    Attributes:
        labels (ClassLabels): The collection of textual class labels.
        vocab (Vocabulary): The collection of all available words.
    """

    def __init__(self, labels, vocab):
        self.labels = labels
        self.vocab = vocab

    @abstractmethod
    def predict(self, test, pool, dbgc):
        """
        Computes a prediction for the specified test phrase, using the
        specified processing pool to improve performance.

        Please note that this function only accepts a single test.  This test
        has the form of a sparse vector, where each index corresponds to a
        word in this learning algorithm's vocabulary and each element at an
        index is the number of times that word appears in the test document.

        :param test: The test to derive a prediction from.
        :param pool: The processing pool to use.
        :param dbgc: The debug console to use.
        :return: A prediction.
        """
        pass

    @abstractmethod
    def test(self, ts, pool, dbgc):
        """
        Computes predictions for any and all tests in the specified testing
        set, using the specified processing pool to improve performance.

        :param ts: The testing set to evaluate.
        :param pool: The processing pool to use.
        :param dbgc: The debug console to use.
        :return: A collection of results ordered by test id.
        """
        pass

    @abstractmethod
    def train(self, tdb, pool, dbgc):
        """
        Trains this learning algorithm with any and all examples in the
        specified training database and uses the specified processing pool to
        improve performance.

        :param tdb: The training database to use.
        :param pool: The processing pool to use.
        :param dbgc: The debug console to use.
        """
        pass

    @abstractmethod
    def validate(self, vds, pool, dbgc):
        """
        Computes predictions for any and all tests in the specified
        validation set and compares them to their expected outcomes,
        using the specified processing pool to improve performance.

        :param vds: The validation set to evaluate.
        :param pool: The processing pool to use.
        :param dbgc: The debug console to use.
        :return: A collection of both experimental and expected results
        ordered by test id.
        """
        pass

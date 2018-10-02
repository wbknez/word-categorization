"""
Contains all classes and functions related to implementing the ore portion of
this project - the actual machine learning algorithms themselves.
"""
import numpy as np
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
    def reset(self, dbgc):
        """
        Resets this learning algorithm to a clean, stable state that will
        allow it to be reused.

        More formally, the post-condition of this function is that this
        learning algorithm is ready for train() to be called without error.

        :param dbgc: The debug console to use.
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


class NaiveBayesLearningAlgorithm(LearningAlgorithm):
    """

    """

    def __init__(self, labels, vocab, beta):
        super().__init__(labels, vocab)

        self.beta = beta
        self.priors = np.full(labels.count + 1, 0.0, dtype=np.float)

    def compute_priors(self, tdb):
        """
        Computes the prior probabilities for each individual document class
        relative to the others.

        Priors are converted into logarithmic form in preparation for training.

        :param tdb: The training database to use.
        """
        classes, frequencies = tdb.get_class_frequencies()
        self.priors[classes] = np.log2(frequencies)

    def predict(self, test, pool, dbgc):
        pass

    def reset(self, dbgc):
        self.priors.fill(0.0)

    def test(self, ts, pool, dbgc):
        pass

    def train(self, tdb, pool, dbgc):
        # Training consists of the following steps:
        #   1) Compute priors.
        #   2) Compute sub-matrices for indexing.
        #   4) Compute word count matrix using sparse vectors.
        dbgc.info("Computing priors P(Yk) for all classes")
        self.compute_priors(tdb)

    def validate(self, vds, pool, dbgc):
        pass

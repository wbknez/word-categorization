"""
Contains all classes and functions related to implementing the ore portion of
this project - the actual machine learning algorithms themselves.
"""
from functools import partial
from itertools import chain

import numpy as np
from abc import ABCMeta, abstractmethod

from wordcat.sparse import SparseMatrix, SparseVector


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
    def predict(self, test, dbgc):
        """
        Computes a prediction for the specified test phrase.

        Please note that this function only accepts a single test.  This test
        has the form of a sparse vector, where each index corresponds to a
        word in this learning algorithm's vocabulary and each element at an
        index is the number of times that word appears in the test document.

        :param test: The test to derive a prediction from.
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

    def test(self, ts, pool, dbgc):
        """
        Computes predictions for any and all tests in the specified testing
        set, using the specified processing pool to improve performance.

        :param ts: The testing set to evaluate.
        :param pool: The processing pool to use.
        :param dbgc: The debug console to use.
        :return: A collection of results ordered by test id.
        """
        predictor = partial(self.predict, dbgc=dbgc)
        results = pool.map(predictor, ts.tests)
        return ts.ids, results

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

        self.alpha = 1 + beta
        self.priors = np.full(labels.count + 1, 0.0, dtype=np.float)
        self.probs = {}

    def compute_map_for_class(self, counts, alpha, V):
        denominator = counts.data[0] + ((alpha - 1) * V)
        counts.data[0] = 0.0

        return SparseVector(np.log2(np.divide(np.add(counts.data, alpha - 1),
                                              denominator)),
                            counts.indices, counts.size)

    def compute_map_estimates(self, pool, word_counts):
        """

        :param pool:
        :param word_counts:
        :return:
        """
        estimator = partial(self.compute_map_for_class, alpha=self.alpha,
                            V=self.vocab.count)
        results = pool.map(estimator, word_counts)
        return {key: value for key, value in enumerate(results, 1)}

    def compute_priors(self, tdb):
        """
        Computes the prior probabilities for each individual document class
        relative to the others.

        Priors are converted into logarithmic form in preparation for training.

        :param tdb: The training database to use.
        """
        classes, frequencies = tdb.get_class_frequencies()
        self.priors[classes] = np.log2(frequencies)

    def compute_sub_matrix_for_class(self, classz, tdb):
        """

        :param classz:
        :param tdb:
        :return:
        """
        class_indices = np.where(tdb.classes == classz)[0]
        row_indices = [np.where(tdb.counts.rows == c)[0] for c in class_indices]

        sub_indices = np.array(list(chain(*row_indices)), copy=False,
                               dtype=np.uint32)
        return SparseMatrix(tdb.counts.data[sub_indices],
                            tdb.counts.rows[sub_indices],
                            tdb.counts.cols[sub_indices],
                            tdb.counts.shape)

    def compute_sub_matrices(self, pool, tdb):
        """

        :return:
        """
        indexer = partial(self.compute_sub_matrix_for_class, tdb=tdb)
        return pool.map(indexer, [i for i in range(1, 21)])

    def compute_word_counts_for_class(self, submat):
        """

        :param submat:
        :return:
        """
        counts = [np.sum(submat.data)]
        indices = [0]

        for i in range(1, submat.shape[1]):
            col_indices = np.where(submat.cols == i)[0]
            count = np.sum(submat.data[col_indices])

            if count != 0:
                counts.append(count)
                indices.append(i)
        return SparseVector(np.array(counts, copy=False,
                                     dtype=np.uint16),
                            np.array(indices, copy=False,
                                     dtype=np.uint32),
                            submat.shape[1])

    def compute_word_counts(self, pool, submats):
        """

        :param pool:
        :param submats:
        :param tdb:
        """
        return pool.map(self.compute_word_counts_for_class, submats)

    def predict(self, test, dbgc):
        max_index = 1
        max_score = -1e15

        for index, _ in self.labels:
            counts = self.probs[index]
            product, remainder = test.multiply(counts)

            score = self.priors[index] + (counts.data[0] * remainder) +\
                    np.sum(product.data)

            if score > max_score:
                max_index = index
                max_score = score
        return max_index

    def reset(self, dbgc):
        self.priors.fill(0.0)
        self.probs.clear()

    def train(self, tdb, pool, dbgc):
        # Training consists of the following steps:
        #   1) Compute priors.
        #   2) Compute sub-matrices for indexing.
        #   4) Compute word count matrix using sparse vectors.
        dbgc.info("Computing priors P(Yk) for all classes.")
        self.compute_priors(tdb)

        dbgc.info("Computing sub-matrices for indexing.")
        submats = self.compute_sub_matrices(pool, tdb)

        dbgc.info("Computing word count matrices (# of Xi in Yk).")
        word_counts = self.compute_word_counts(pool, submats)

        dbgc.info("Computing MAP table P(Xi|Yk) using alpha of {:.2f}.",
                  self.alpha)
        self.probs = self.compute_map_estimates(pool, word_counts)

    def validate(self, vds, pool, dbgc):
        pass

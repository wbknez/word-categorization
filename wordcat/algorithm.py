"""
Contains all classes and functions related to implementing the ore portion of
this project - the actual machine learning algorithms themselves.
"""
from functools import partial

import numpy as np
from abc import ABCMeta, abstractmethod

from wordcat.sparse import SparseVector
from wordcat.storage import Prediction


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

    def test(self, pool, ts, dbgc):
        """
        Computes predictions for any and all tests in the specified testing
        set, using the specified processing pool to improve performance.

        :param pool: The processing pool to use.
        :param ts: The testing set to evaluate.
        :param dbgc: The debug console to use.
        :return: A collection of predictions ordered by test id.
        """
        predictor = partial(self.predict, dbgc=dbgc)
        return pool.map(predictor, ts.tests)

    @abstractmethod
    def train(self, pool, tdb, params, dbgc):
        """
        Trains this learning algorithm with any and all examples in the
        specified training database and uses the specified processing pool to
        improve performance.

        :param pool: The processing pool to use.
        :param tdb: The training database to use.
        :param params: The user-chosen parameters to use.
        :param dbgc: The debug console to use.
        """
        pass

    def validate(self, pool, vds, dbgc):
        """
        Computes predictions for any and all tests in the specified
        validation set and compares them to their expected outcomes,
        using the specified processing pool to improve performance.

        :param pool: The processing pool to use.
        :param vds: The validation set to evaluate.
        :param dbgc: The debug console to use.
        :return: A collection of both experimental and expected results
        ordered by test id.
        """
        pass


class LogisticRegressionLearningAlgorithm(LearningAlgorithm):
    """


    Attributes:
        norms (np.array):
        w (np.array):
    """

    def __init__(self, labels, vocab):
        super().__init__(labels, vocab)

        self.norms = None
        self.w = None

    def normalize(self, obj, norms=None):
        """
        Normalizes the specified NumPy array or matrix (of arrays) using the
        specified normalizing constants if supplied, otherwise the column
        summations are used instead.

        :param obj: The NumPy object to normalize
        :param norms: The normalizing constants to use, if any.  If None then
        the column summations will be used instead.
        :return: A normalized NumPy object.
        """
        if not norms:
            norms = obj.sum(axis=0)
        indices = np.where(norms != 0)

        obj[:, indices] = obj[:, indices] / norms[indices]
        return obj

    def predict(self, test, dbgc):
        dbgc.info("Working on test: {}".format(test.id))

        norm_query = self.normalize(test.query.to_dense(np.float32), self.norms)
        scores = np.dot(self.w, norm_query.T)

        dbgc.info("Scores for test: {} are:\n{}.".format(
            test.id,
            ["{:.2f}".format(score) for score in scores]
        ))

        return Prediction(test.id, np.argmax(scores[1:]) + 1)

    def train(self, pool, tdb, params, dbgc):
        eta = params.eta
        k = np.max(tdb.classes) + 1
        l = params.lambda_
        m = tdb.row_count
        n = tdb.col_count

        dbgc.info("Converting training data to dense format.")
        x = tdb.counts.to_dense(np.float32)

        dbgc.info("Normalzing training data.")
        self.norms = x.sum(axis=0)
        x = self.normalize(x, self.norms)

        dbgc.info("Creating initial weights matrix W(0).")
        self.w = np.random.random((k, n))
        self.w[0, :] = 0.0
        self.w[:, 0] = 0.0

        dbgc.info("Normalizing weights.")
        self.w = self.normalize(self.w, self.w.sum(axis=0))

        dbgc.info("Creating initial probability matrix P(Y|W,X).")
        p_ywx = np.zeros((k, m), dtype=np.float32)

        dbgc.info("Compiling deltas")
        deltas = tdb.create_deltas().astype(np.float32)

        for i in range(params.steps):
            dbgc.info("Working on iteration: {} of {}.".format(i, params.steps))

            dbgc.info("Calculating probability matrix P(Yk|W,Xi).")
            p_ywx = np.exp(np.dot(self.w, x.T))
            p_ywx[-1, :] = 1.0

            dbgc.info("Normalizing probability matrix.")
            p_ywx = self.normalize(p_ywx, p_ywx.sum(axis=0))

            dbgc.info("Computing new weights W(t+1) using current W(t).")
            self.w = self.w + eta * (np.dot((deltas - p_ywx), x) - (l * self.w))
            self.w = self.normalize(self.w, self.w.sum(axis=0))


class NaiveBayesLearningAlgorithm(LearningAlgorithm):
    """


    Attributes:
        maps (dict): The mapping of maximum apriori estimates by class.
        priors (list): The list of prior probabilitys per class.
    """

    def __init__(self, labels, vocab):
        super().__init__(labels, vocab)

        self.maps = {}
        self.priors = {}

    def count_words(self, mat):
        """
        Counts the total number of words in each feature in the specified
        sparse matrix.

        :param mat: The sparse matrix of word counts to use.
        :return: A tuple containing the total number of words in an entire
        matrix, and a sparse vector of resulting column sums.
        """
        return mat.sum(), SparseVector.from_list(
            [column.sum() for column in mat.get_columns()], np.uint32
        )

    def find_max(self, scores):
        """
        Determines the class of the maximum score.

        If there is a tie, then the class with the largest prior wins.  If
        there are two classes with the same score and prior, then the
        lexographically smallest class wins.

        :param scores: The list of scores to use.
        :return: The maximum score's class.
        """
        max_score = max(scores.values())
        max_keys = [
            key for key in sorted(scores.keys()) if scores[key] == max_score
        ]

        if len(max_keys) == 1:
            return max_keys[0]
        return max_keys[[self.priors[key] for key in max_keys].index(max_score)]

    def make_map(self, counts, alpha, V):
        """
        Computes a MAP estimate for the specified matrix of word counts with
        the specified alpha and vocabulary length.

        :param counts: The matrix to compute MAP estimates for.
        :param alpha: The alpha Laplace coefficient to use.
        :param V: The size of the vocabulary to use.
        :return: A tuple consisting of the base MAP estimate when the count
        for an arbitrary word is zero, and a sparse vector of MAP estimates
        for every word in a vocabulary.
        """
        denominator = 1.0 / (counts[0] + ((alpha - 1) * V))
        return (
            np.log2((alpha - 1) * denominator),
            ((counts[1] + (alpha - 1)) * denominator).log2()
        )

    def predict(self, test, dbgc):
        scores = {}

        dbgc.info("Working on test: {}".format(test.id))

        for classz, prior in self.priors.items():
            dbgc.info("Computing P(Ynew) for class: {} for test: {}".format(
                classz, test.id
            ))

            no_words, words = self.maps[classz]
            intersect, diff = test.query.venn(words)

            scores[classz] =\
                prior + (intersect * words).sum() + (diff * no_words).sum()

        dbgc.info("Scores for test: {} are:\n{}.".format(
            test.id,
            ["{:.2f}".format(score) for score in scores.values()]
        ))
        return Prediction(test.id, self.find_max(scores))

    def train(self, pool, tdb, params, dbgc):
        dbgc.info("Calculating priors P(Yk) for all classes.")
        self.priors = {
            k: np.log2(v) for k, v in tdb.create_frequencies().items()
        }

        dbgc.info("Creating sub-matrices for indexing.")
        submats = pool.map(
            tdb.select, [classz for classz in sorted(self.priors.keys())]
        )

        dbgc.info("Counting words in sub-matrices (# of Xi in Yk).")
        word_counts = pool.map(self.count_words, submats)

        dbgc.info("Computing MAP table P(Xi|Yk) using alpha of {:.2f}.",
                  1 + params.beta)
        y_k, x_k = zip(*pool.map(
            partial(self.make_map, alpha=1 + params.beta, V=self.vocab.count),
            word_counts
        ))

        dbgc.info("Compressing MAP table into learner cache.")
        self.maps = {
            cls: [y_k[idx], x_k[idx]] \
                for idx, cls in enumerate(sorted(self.priors.keys()))
        }

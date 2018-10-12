"""
Contains all classes and functions related to implementing the ore portion of
this project - the actual machine learning algorithms themselves.
"""
from functools import partial

import numpy as np
from abc import ABCMeta, abstractmethod

from wordcat.sparse import SparseVector


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

        :param mat:
        :return:
        """
        return mat.sum(), SparseVector.from_list(
            [column.sum() for column in mat.get_columns()], np.uint32
        )

    def find_max(self, scores):
        """

        :param scores:
        :return:
        """
        max_score = max(scores.values())
        max_keys = [
            key for key in sorted(scores.keys()) if scores[key] == max_score
        ]

        if len(max_keys) == 1:
            return max_keys[0]
        return max_keys[[self.priors[key] for key in max_keys].index(max_score)]

    def make_map(self, counts, alpha, V):
        denom = 1.0 / (counts[0] + ((alpha - 1) * V))
        return (
            np.log2((alpha - 1) * denom),
            ((counts[1] + (alpha - 1)) * denom).log2()
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
        return self.find_max(scores)

    def train(self, pool, tdb, params, dbgc):
        dbgc.info("Calculating priors P(Yk) for all classes.")
        self.priors = {
            k: np.log2(v) for k, v in tdb.create_class_frequency_table().items()
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

    def validate(self, vds, pool, dbgc):
        pass

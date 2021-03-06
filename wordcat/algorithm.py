"""
Contains all classes and functions related to implementing the ore portion of
this project - the actual machine learning algorithms themselves.
"""
from functools import partial

import numpy as np
from abc import ABCMeta, abstractmethod

from wordcat.sparse import SparseVector
from wordcat.storage import Prediction, ValidatedPrediction, Ranking


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

    def validate(self, pool, ts, ans, dbgc):
        """
        Computes predictions for any and all tests in the specified
        validation set and compares them to their expected outcomes,
        using the specified processing pool to improve performance.

        :param pool: The processing pool to use.
        :param ts: The testing set to evaluate.
        :param ans: The correct predictive results.
        :param dbgc: The debug console to use.
        :return: A collection of both experimental and expected results
        ordered by test id.
        """
        predictions = self.test(pool, ts, dbgc)
        return [
            ValidatedPrediction(ans[i], predictions[i]) for i in range(len(ts))
        ]


class LogisticRegressionLearningAlgorithm(LearningAlgorithm):
    """
    Represents an implementation of LearningAlgorithm that attempts to
    classify documents into categories using a logistic regression
    classification algorithm.

    Attributes:
        norms (np.array): The column sums used to normalize the training data.
        w (np.array): The list of learned weights.
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
        if np.any(norms) is None:
            print("Not any!")
            norms = obj.sum(axis=0)
        indices = np.where(norms != 0)

        obj[:, indices] = obj[:, indices] / norms[indices]
        return obj

    def predict(self, test, dbgc):
        dbgc.info("Working on test: {}".format(test.id))
        indices = np.where(self.norms != 0)

        norm_query = test.query.to_dense(np.float32)
        norm_query[indices] = norm_query[indices] / self.norms[indices]

        scores = np.dot(self.w, norm_query.T)

        dbgc.info("Scores for test: {} are:\n{}.".format(
            test.id,
            ["{:.2f}".format(score) for score in scores]
        ))

        return Prediction(test.id, np.argmax(scores))

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

            dbgc.info("Normalizing new weights W(t + 1).")
            self.w = self.normalize(self.w, self.w.sum(axis=0))


class NaiveBayesLearningAlgorithm(LearningAlgorithm):
    """
    Represents an implementation of LearningAlgorithm that attempts to
    classify documents into categories using a naive Bayes classification
    algorithm.

    Attributes:
        maps (dict): The mapping of maximum apriori estimates by class.
        priors (dict): The mapping of prior probabilitys to class.
        rankings (list): The list of the top n most important words.
    """

    def __init__(self, labels, vocab):
        super().__init__(labels, vocab)

        self.maps = {}
        self.priors = {}
        self.rankings = []

    def compute_tfidf(self, word, freqs, labels, maps, off_set):
        """
        Computes the term frequency inverse document frequency value for the
        specified word using the specified frequencies, labels,
        MAP estimates.

        The off-set is used to prevent divide-by-zero error(s) when computing
        the IDF term.

        :param word: The word to compute the TF-IDF for.
        :param freqs: The list of word frequencies to use.
        :param labels: The class labels to use.
        :param maps: The MAP estimates to use.
        :param off_set: The off set to use to prevent division by zero errors.
        :return: The TF-IDF value.
        """
        if word == 0:
            return -1e10

        values = np.exp(np.array([maps[c][1].value_at(word) for c, _ in labels],
                                 dtype=np.float32)) + off_set
        freq = freqs[word]

        if freq == 0.0:
            freq = 1.0

        return -np.log2(freq) * np.sum(np.log2(values))

    def compute_word_frequency(self, column, total):
        """
        Returns the frequency of the specified word, given as a single column
        in a training set.

        The formula for the frequency is:
            f(w) =    # of occurences of word w
                   --------------------------------
                   # of total words in training set

        :param column: The word to compute the frequency for.
        :param total: The total number of words in a training set.
        :return: The word frequency.
        """
        return column.sum() / total

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

    def rank_words(self, tfidfs, count):
        """
        Sorts and compiles a list of the specified count length of the most
        important words using the specified TF-IDF values.

        :param tfidfs: The collection of TF-IDF values to use.
        :param count: The number of words to return.
        :return: A list of word rankings.
        """
        array = np.array(tfidfs, dtype=np.float32)
        indices = np.argsort(array)[::-1][:count]

        return [Ranking(self.vocab[i], array[i]) for i in indices]

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


        if params.rank_words:
            dbgc.info("Computing P(Xi) for all words in dataset.")
            freqs = pool.map(
                partial(self.compute_word_frequency, total=tdb.counts.sum()),
                tdb.counts.get_columns()
            )

            dbgc.info("Calculating TF-IDF.")
            tfidfs = pool.map(
                partial(
                    self.compute_tfidf, freqs=freqs, labels=self.labels,
                    maps=self.maps, off_set=1
                ),
                [i for i in range(tdb.counts.col_count)]
            )

            dbgc.info("Calculating word rankings.")
            self.rankings = self.rank_words(tfidfs, params.rank_count)

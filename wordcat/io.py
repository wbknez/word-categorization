"""
Contains all classes and functions relating to I/O operations for this
project; this includes, but is not limited to, reading and writing objects
from both CSV and sparse (compressed) data formats.
"""
import csv
import pickle

import numpy as np
from collections import namedtuple
from functools import partial

from wordcat.sparse import SparseMatrix, SparseVector
from wordcat.storage import TrainingDatabase, TestingSet, Vocabulary, \
    ClassLabels


class CsvIO:
    """
    Represents a collection of utility methods for working with comma
    separated values, or CSV, files.
    """

    class Line(namedtuple("Line", ["classz", "cols", "data", "id"])):
        """
        Represents a single, sparse row of CSV data from an external source.
        """

        pass

    @staticmethod
    def generate_lines(stream, delimiter=","):
        """
        Parses the specified CSV stream, whose values are separated by the
        specified delimiter, and returns a generator over each row index and
        accompanying line.

        :param stream: The CSV stream to use.
        :param delimiter: The character that separates values in the stream.
        :return: A generator over each parsed line.
        """
        reader = csv.reader(stream, delimiter=delimiter)
        for line in reader:
            yield line

    @staticmethod
    def process_line(line, skip_class, skip_id):
        """
        Processes the specified line of CSV data from the specified row index
        into a sparse list.

        The total number of elements depends on whether or not the final
        element represents a class (i.e. is a result).  For this project,
        the first element (i.e. index of zero) is skipped regardless as it
        denotes a unique ID.

        This function manages no state as it is intended to be used in
        parallel to improve file parsing performance.

        :param line: The line of CSV data to parse.
        :param skip_class: Whether or not the last element of a line is
        a classification marker and should be ignored.
        :param skip_id: Whether or not the first element of a line is a
        unique identifier and should be ignored.
        :return: A sparse representation of a line of CSV data.
        """
        classz = None if skip_class else int(line[-1])
        cols = []
        data = []
        id = None if skip_id else int(line[0])
        last_index = len(line) - (1 if not skip_class else 0)

        for i in range(1, last_index):
            if line[i] != "0":
                cols.append(i)
                data.append(int(line[i]))

        return CsvIO.Line(classz, cols, data, id)

    @staticmethod
    def read_database(pool, stream):
        """
        Reads a training database from the specified CSV stream using the
        specified processing pool to improve I/O performance.

        :param pool: The processing pool to use.
        :param stream: The CSV stream to read from.
        :return: A database filled with data to train on.
        """
        classes = []
        cols = []
        data = []
        rows = []

        processor = partial(CsvIO.process_line, skip_class=False, skip_id=True)
        results = pool.map(processor, CsvIO.generate_lines(stream))

        for index, result in enumerate(results):
            classes.append(result.classz)
            cols.extend(result.cols)
            data.extend(result.data)
            rows.extend([index] * len(result.cols))

        return TrainingDatabase(
            np.array(classes, copy=False, dtype=np.uint8),
            SparseMatrix(np.array(data, copy=False, dtype=np.uint16),
                         np.array(rows, copy=False, dtype=np.uint32),
                         np.array(cols, copy=False, dtype=np.uint32),
                         (max(rows) + 1, max(cols) + 1))
        )

    @staticmethod
    def read_labels(_, stream):
        """
        Reads a collection of class labels from the specified CSV stream.

        :param _: The processing pool to use (unused).
        :param stream: The CSV stream to read from.
        :return: A collection of class labels.
        """
        classes = {}

        for line in CsvIO.generate_lines(stream, delimiter=" "):
            classes[int(line[0])] = line[1]
        return ClassLabels(classes)

    @staticmethod
    def read_set(pool, stream):
        """
        Resds a testing set from the specified CSV stream using the specified
        processing pool to improve I/O performance.

        :param pool: The processing pool to use.
        :param stream: The CSV stream to read from.
        :return: A set filled with some data to test with.
        """
        ids = []
        tests = []

        processor = partial(CsvIO.process_line, skip_class=True, skip_id=False)
        results = pool.map(processor, CsvIO.generate_lines(stream))

        for result in results:
            ids.append(result.id)
            tests.append(SparseVector(data=np.array(result.data, copy=False,
                                               dtype=np.uint16),
                                      indices=np.array(result.cols, copy=False,
                                               dtype=np.uint32),
                                      size=61189))

        return TestingSet(ids, tests)

    @staticmethod
    def read_vocabulary(_, stream):
        """
        Reads a vocabulary set from the specified CSV stream.

        :param _: The processing pool to use (unused).
        :param stream: The CSV stream to read from.
        :return: A collection of words to use as a vocabulary.
        """
        words = []

        for line in CsvIO.generate_lines(stream):
            words.append(line)
        return Vocabulary(words)

    @staticmethod
    def write_predictions(stream, header, predictions):
        """
        Writes the specified predictions, associated and sorted by id, to the
        specified CSV stream with the specified header.

        :param stream: The CSV stream to write to.
        :param header: The label header(s) to use.
        :param predictions: The collection of predictions to use.
        """
        writer = csv.writer(stream, delimiter=",", quoting=csv.QUOTE_NONE)

        writer.writerow(header)
        for prediction in predictions:
            writer.writerow([prediction.id, prediction.result])


class PickleIO:
    """
    Represents a collection of utility methods for working with Pickle files
    for reading and storing objects in this project quickly and efficiently.
    """

    @staticmethod
    def read_database(stream):
        """
        Reads a training database from the specified Pickle stream.

        :param stream: The Pickle stream to read from.
        :return: A training database created from sparse data.
        """
        return pickle.load(stream)

    @staticmethod
    def read_labels(stream):
        """
        Reads a collection of class labels from the specified Pickle stream.

        :param stream: The Pickle stream to read from.
        :return: A pre-created collection of class labels.
        """
        return pickle.load(stream)

    @staticmethod
    def read_set(stream):
        """
        Reads a testing set from the specified Pickle stream.

        :param stream: The Pickle stream to read from.
        :return: A testing set created from sparse data.
        """
        return pickle.load(stream)

    @staticmethod
    def read_vocabulary(stream):
        """
        Reads a vocabulary set from the specified Pickle stream.

        :param stream: The Pickle stream to read from.
        :return: A pre-created vocabulary.
        """
        return pickle.load(stream)

    @staticmethod
    def write_database(stream, tdb):
        """
        Writes the specified database to the specified Pickle stream.

        :param stream: The Pickle stream to write to.
        :param tdb: The database to write.
        """
        pickle.dump(tdb, stream, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def write_labels(stream, labels):
        """
        Writes the specified class labels to the specified Pickle stream.

        :param stream: The pickle stream to write to .
        :param labels: The class labels to write.
        """
        pickle.dump(labels, stream, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def write_set(stream, ts):
        """
        Writes the specified testing set to the specified Pickle stream.

        :param stream: The pickle stream to write to.
        :param ts: The testing set to write.
        """
        pickle.dump(ts, stream, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def write_vocabulary(stream, vocab):
        """
        Writes the specified vocabulary set to the specified Pickle stream.

        :param stream: The pickle stream to write to.
        :param vocab: The vocabulary to write.
        """
        pickle.dump(vocab, stream, pickle.HIGHEST_PROTOCOL)

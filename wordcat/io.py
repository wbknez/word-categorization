"""

"""
import csv
import numpy as np
from collections import namedtuple
from functools import partial

from wordcat.sparse import SparseMatrix
from wordcat.storage import TrainingDatabase


class CsvIO:
    """

    """

    class Line(namedtuple("Line", ["row", "cols", "data"])):
        """

        Attributes:

        """

        pass

    @staticmethod
    def generate_lines(stream, delimiter=","):
        """

        :param stream:
        :param delimiter:
        :return:
        """
        reader = csv.reader(stream, delimiter=delimiter)
        for row, line in enumerate(reader):
            yield row, line

    @staticmethod
    def process_line(line, row, skip_class):
        """

        :param line:
        :param row:
        :param skip_class:
        :return:
        """
        cols = []
        data = []
        last_index = len(line) - (1 if not skip_class else 0)

        for i in range(1, last_index):
            if line[i] != "0":
                cols.append(i)
                data.append(int(line[i]))

        return CsvIO.Line(row, cols, data)

    @staticmethod
    def read_database(pool, stream):
        """

        :param pool:
        :param stream:
        :return:
        """
        classes = []
        cols = []
        data = []
        rows = []

        procs = []
        results = []

        for row, line in CsvIO.generate_lines(stream):
            classes.append(int(line[-1]))

            processor = partial(CsvIO.process_line, row=row, skip_class=False)
            procs.append(pool.map_async(processor, (line, )))
        for proc in procs:
            results.append(proc.get()[0])

        # Sort by row.
        results.sort(key=lambda t: t[0])

        for result in results:
            cols.extend(result[1])
            data.extend(result[2])
            rows.extend([result[0]] * len(result[1]))

        return SparseMatrix(np.array(data, copy=False, dtype=np.uint16),
                            np.array(rows, copy=False, dtype=np.uint32),
                            np.array(cols, copy=False, dtype=np.uint32),
                            shape=(max(rows) + 1, max(cols) + 2))


class SparseIO:
    """

    """

    @staticmethod
    def read_database(stream):
        """


        :param stream: The file stream to read from.
        :return: A training database created from a sparse matrix.
        """
        arrays = np.load(stream)
        return TrainingDatabase(arrays["classes"],
                                SparseMatrix(arrays["data"],
                                             arrays["rows"],
                                             arrays["cols"],
                                             tuple(arrays["shape"])))

    @staticmethod
    def read_set(stream):
        """

        :param stream:
        :return:
        """
        pass

    @staticmethod
    def write_database(stream, tdb):
        """

        :param stream:
        :param tdb:
        :return:
        """
        np.savez(stream, data=tdb.counts.data, rows=tdb.counts.rows,
                 cols=tdb.counts.cols, classes=tdb.classes,
                 shape=tdb.counts.shape)

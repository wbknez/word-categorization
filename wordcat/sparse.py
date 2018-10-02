"""
Contains classes and functions pertaining to sparse data structures.
"""
import numpy as np


class SparseMatrix:
    """
    Represents a matrix that only stores non-zero elements, making it memory
    efficient for managing large amounts of sparse tabular data.

    Attributes:
        cols (np.array): The array of column indices.
        data (np.array): The array of non-zero elements.
        rows (np.array): The array of row indices.
        shape (tuple): The "true" dimensions in (row, column) form.
    """

    def __init__(self, data, rows, cols, shape):
        if rows.size != cols.size:
            raise ValueError("The size of the indice arrays must be "
                             "equivalent in both directions: rows is {} but "
                             "cols is {}".format(rows.size, cols.size))
        if cols.size != data.size:
            raise ValueError("The size of the data array must equal that of "
                             "the indices: "
                             "{} instead of {}.".format(data.size, cols.size))
        if not np.issubdtype(cols.dtype, np.integer):
            raise ValueError("Column indices are not of integral type: "
                             "{}".format(cols.dtype))
        if not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("Row indices are not of integral type: "
                             "{}".format(rows.dtype))

        self.cols = cols
        self.data = data
        self.rows = rows
        self.shape = shape

    def __eq__(self, other):
        if isinstance(other, SparseMatrix):
            return np.array_equal(self.cols, other.cols) and \
                   np.array_equal(self.data, other.data) and \
                   np.array_equal(self.rows, other.rows) and \
                   self.shape == other.shape
        return NotImplemented

    def __getattr__(self, item):
        if item == "dtype":
            return self.data.dtype

    def __getstate__(self):
        return self.__dict__.copy()

    def __len__(self):
        """
        Returns the number of non-zero elements in this sparse matrix.

        Please note that the length is not the same as the size.

        :return: The number of non-zero elements.
        """
        return self.data.size

    def __ne__(self, other):
        return not self == other

    def __setstate__(self, state):
        self.__dict__.update(state)

    def get_column(self, index):
        """
        Returns a data-only view of this sparse matrix located at the
        specified column index.

        Please note that the resulting data array is stacked horizontally,
        not vertically, despite describing a column.

        :param index: The column index to use.
        :return: A column of data at an index.
        :raise IndexError: If the index is out of bounds.
        """
        if index < 0 or index >= self.shape[1]:
            raise IndexError("Column index is out of bounds: {}".format(index))
        return self.data[np.where(self.cols == index)]

    def get_row(self, index):
        """
        Returns a data-only view of this sparse matrix located at the
        specified row index.

        :param index: The row index to use.
        :return: A row of data at an index.
        :raise IndexError: If the index is out of bounds.
        """
        if index < 0 or index >= self.shape[0]:
            raise IndexError("Row index is out of bounds: {}".format(index))
        return self.data[np.where(self.rows == index)]

    @staticmethod
    def from_list(dense_matrix):
        """
        Creates a sparse matrix from the specified nested (two-dimensional)
        list of elements.

        The list of lists denotes a matrix in row-major order.

        :param dense_matrix: The list of elements to use.
        :return: A new sparse matrix.
        """

        cols = []
        data = []
        rows = []

        for row_index, row in enumerate(dense_matrix):
            for col_index, col in enumerate(row):
                if col != 0:
                    cols.append(col_index)
                    data.append(col)
                    rows.append(row_index)

        return SparseMatrix(np.array(data), np.array(rows, np.int),
                            np.array(cols, np.int),
                            (len(dense_matrix), len(dense_matrix[0])))


class SparseVector:
    """
    Represents a matrix that only stores non-zero elements, making it memory
    efficient for managing large amounts of sparse sequential data.

    Attributes:
        data (np.array): The array of non-zero elements.
        indices (np.array): The array of indices.
        size (int): The maximum number of potential non-zero elements.
    """

    def __init__(self, data, indices, size):
        if data.size != indices.size:
            raise ValueError("The number of data elements and the size of the "
                             "indice array must match: "
                             "{} instead of {}.".format(data.size,
                                                        indices.size))
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Indices are not of integral type: "
                             "{}".format(indices.dtype))

        self.data = data
        self.indices = indices
        self.size = size

    def __eq__(self, other):
        if isinstance(other, SparseVector):
            return np.array_equal(self.data, other.data) and \
                   np.array_equal(self.indices, other.indices)
        return NotImplemented

    def __getattr__(self, item):
        if item == "dtype":
            return self.data.dtype

    def __getstate__(self):
        return self.__dict__.copy()

    def __len__(self):
        return self.data.size

    def __ne__(self, other):
        return not self == other

    def __setstate__(self, state):
        self.__dict__.update(state)

"""
Contains classes and functions pertaining to sparse data structures.
"""
from copy import copy

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

    def __copy__(self):
        return SparseMatrix(
            np.copy(self.data), np.copy(self.rows), np.copy(self.cols),
            copy(self.shape)
        )

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

    def __getitem__(self, item):
        return self.data[item], (self.rows[item], self.cols[item])

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
        Returns the column at the specified index in this sparse matrix as a
        sparse vector whose indices represent valid rows.

        :param index: The column index to use.
        :return: A sparse vector of column data at an index.
        :raise IndexError: If the index is out of bounds.
        """
        if index < 0 or index >= self.shape[1]:
            raise IndexError("Column index is out of bounds: {}".format(index))

        indices = np.where(self.cols == index)
        return SparseVector(self.data[indices], self.rows[indices],
                            self.shape[0])

    def get_columns(self):
        """
        Returns a generator over the collection of columns in this sparse
        matrix.

        :return: A collection of columns.
        """
        for i in range(self.shape[1]):
            yield self.get_column(i)

    def get_row(self, index):
        """
        Returns the row at the specified index in this sparse matrix as a
        sparse vector whose indices represent valid columns.

        :param index: The row index to use.
        :return: A sparse vector of row data at an index.
        :raise IndexError: If the index is out of bounds.
        """
        if index < 0 or index >= self.shape[0]:
            raise IndexError("Row index is out of bounds: {}".format(index))

        indices = np.where(self.rows == index)[0]
        return SparseVector(self.data[indices], self.cols[indices],
                            self.shape[1])

    def get_rows(self):
        """
        Returns a generator over the collection of columns in this sparse
        matrix.

        :return: A collection of columns.
        """
        for i in range(self.shape[0]):
            yield self.get_row(i)

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

    def __copy__(self):
        return SparseVector(np.copy(self.data), np.copy(self.indices),
                            self.size)

    def __eq__(self, other):
        if isinstance(other, SparseVector):
            return np.array_equal(self.data, other.data) and \
                   np.array_equal(self.indices, other.indices)
        return NotImplemented

    def __getattr__(self, item):
        if item == "dtype":
            return self.data.dtype

    def __getitem__(self, item):
        return self.data[item], self.indices[item]

    def __getstate__(self):
        return self.__dict__.copy()

    def __len__(self):
        return self.data.size

    def __ne__(self, other):
        return not self == other

    def __setstate__(self, state):
        self.__dict__.update(state)

    def multiply(self, vec):
        """
        Multiplies this sparse vector with the specified vector, returning
        both the product via indice intersection as well as the non-applied
        indices as a remainder.

        :param vec: The sparse vector to multiply with.
        :return: The product as an indice intersection and the remaining
        indices that were not used in the calculation.
        """
        if self.size != vec.size:
            raise ValueError("Vector sizes must match in order to multiply: "
                             "{} is not {}.".format(self.size, vec.size))

        my_indices = np.in1d(self.indices, vec.indices)
        vec_indices = np.in1d(vec.indices, self.indices)

        return SparseVector(np.multiply(self.data[my_indices],
                                        vec.data[vec_indices]),
                            self.indices[my_indices], np.size(my_indices)), \
               np.setdiff1d(self.indices, self.indices[my_indices],
                            assume_unique=True)

    def scale(self, scalar):
        """
        Computes the sparse vector that results from scaling this one with
        by the specified amount.

        :param scalar: The (singular) amount to scale by.
        :return: A scaled sparse vector.
        """
        return SparseVector(
            np.multiply(self.data, scalar),
            self.indices,
            self.size
        )

    def slice(self, indices):
        """
        Slices this sparse vector using the specified indices.

        Please note that this is not a typical slicing operation.  This
        function returns a sparse vector with the following properties:
            - it has the same size as the original vector, and
            - its data and indices are direct slices of the original vector.
        Thus, this operation may be considered a sub-vectorization with
        post-conditions.

        :param indices: The indices to slicy by.
        :return: A sliced sparse vector.
        """
        return SparseVector(
            self.data[indices], self.indices[indices], self.size
        )

    def sum(self):
        """
        Computes the sum of the data elements of this sparse vector.

        :return: The sum.
        """
        return np.sum(self.data)

    @staticmethod
    def from_list(dense_list):
        """
        Creates a sparse vector from the specified list of elements.

        :param dense_list: The list of elements to use.
        :return: A new sparse vector.
        """
        data = []
        indices = []

        for index, element in enumerate(dense_list):
            if element != 0:
                data.append(element)
                indices.append(index)

        return SparseVector(np.array(data, copy=False, dtype=np.uint16),
                            np.array(indices, copy=False, dtype=np.uint32),
                            size=len(dense_list))

    @staticmethod
    def from_lists(data, indices, size, dtype=None):
        """
        Creates a sparse vector from the specified lists of elements and
        indices with the specified total size.

        :param data: The list of elements to use.
        :param indices: The list of indices to use.
        :param size: The total size to use.
        :return: A new sparse vector.
        """
        return SparseVector(
            np.array(data, copy=False, dtype=dtype),
            np.array(indices, copy=False, dtype=np.uint32),
            size
        )

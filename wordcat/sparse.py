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

    def sum(self):
        """
        Computes the sum of the data elements of this sparse matrix.

        :return: The sum.
        """
        return np.sum(self.data)

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

    Please note that all mathematical operations on sparse vectors are:
        1. performed element-wise, and
        2. immutable with respect to the operands.
    That is, all mathematical operations return copies of the underlying
    numeric and indice data, making each new sparse vector unique and all
    sparse vectors immutable with respect to the operations performed upon them.

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

    def __add__(self, other):
        if isinstance(other, (float, int)):
            other = SparseVector(np.full(self.data.size, other,
                                         dtype=self.data.dtype),
                                 self.indices, self.size)

        try:
            if self.size != other.size:
                raise ValueError("Vector sizes must be equal for mathematical "
                                 "operations.")

            src_idx = np.in1d(self.indices, other.indices)
            dest_idx = np.in1d(other.indices, self.indices)

            result = SparseVector(
                np.add(self.data[src_idx], other.data[dest_idx]),
                self.indices[src_idx],
                self.size
            )
            result.compact()

            return result

        except TypeError:
            return NotImplemented

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

    def __hash__(self):
        return hash((self.data, self.indices, self.size))

    def __iter__(self):
        for i in range(self.data.size):
            yield self.data[i], self.indices[i]

    def __len__(self):
        """
        Returns the number of non-zero elements in this sparse vector.

        Please note that this value is inherently different than that
        returned by "size".

        :return: The number of non-zero elements.
        """
        return self.data.size

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            other = SparseVector(np.full(self.data.size, other,
                                         dtype=self.data.dtype),
                                 self.indices, self.size)

        try:
            if self.size != other.size:
                raise ValueError("Vector sizes must be equal for mathematical "
                                 "operations.")

            src_idx = np.in1d(self.indices, other.indices)
            dest_idx = np.in1d(other.indices, self.indices)

            result = SparseVector(
                np.multiply(self.data[src_idx], other.data[dest_idx]),
                self.indices[src_idx],
                self.size
            )
            result.compact()

            return result

        except TypeError:
            return NotImplemented

    def __ne__(self, other):
        return not self == other

    def __neg__(self):
        return SparseVector(np.negative(self.data), self.indices, self.size)

    def __repr__(self):
        return "{}({},{},{})".format(self.__class__, repr(self.data),
                                     repr(self.indices), repr(self.size))

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return "({}, {}, {})".format(self.data, self.indices, self.size)

    def __sub__(self, other):
        if isinstance(other, (float, int)):
            other = SparseVector(np.full(self.data.size, other,
                                         dtype=self.data.dtype),
                                 self.indices, self.size)

        try:
            if self.size != other.size:
                raise ValueError("Vector sizes must be equal for mathematical "
                                 "operations.")

            src_idx = np.in1d(self.indices, other.indices)
            dest_idx = np.in1d(other.indices, self.indices)

            result = SparseVector(
                np.subtract(self.data[src_idx], other.data[dest_idx]),
                self.indices[src_idx],
                self.size
            )
            result.compact()

            return result

        except TypeError:
            return NotImplemented

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError("Sparse vectors cannot divide by zero.")

        if isinstance(other, (float, int)):
            other = SparseVector(np.full(self.data.size, other,
                                         dtype=self.data.dtype),
                                 self.indices, self.size)

        try:
            if self.size != other.size:
                raise ValueError("Vector sizes must be equal for mathematical "
                                 "operations.")

            src_idx = np.in1d(self.indices, other.indices)
            dest_idx = np.in1d(other.indices, self.indices)

            return SparseVector(
                np.divide(self.data[src_idx], other.data[dest_idx]),
                self.indices[src_idx],
                self.size
            )

        except TypeError:
            return NotImplemented

    def abs(self):
        """
        Computes the sparse vector that results from taking the absolute
        value of this one.

        :return: The absolute value of a sparse vector.
        """
        return SparseVector(np.abs(self.data), self.indices, self.size)

    def compact(self):
        """

        :return:
        """
        zero_idx = np.where(self.data == 0)
        self.data = np.delete(self.data, zero_idx)
        self.indices = np.delete(self.indices, zero_idx)

    def exp(self):
        """
        Computes the sparse vector that results from exponentiating
        this one.

        :return: The exp of a sparse vector.
        """
        return SparseVector(np.exp(self.data), self.indices, self.size)

    def log2(self):
        """
        Computes the sparse vector that results from taking the base-2
        logarithm of this one.

        :return: The base-2 logarithm of a sparse vector.
        """
        return SparseVector(np.log2(self.data), self.indices, self.size)

    def power(self, a):
        """
        Computes the sparse vector that results from taking this one and
        raising it to the specified power.

        :param a: The amount to raise.
        :return: A sparse vector raised to a power.
        """
        return SparseVector(np.power(self.data, a), self.indices, self.size)

    def sum(self):
        """
        Computes the sum of the data elements of this sparse vector.

        :return: The sum of a sparse vector.
        """
        return np.sum(self.data)

    def venn(self, other):
        """
        Computes both the set intersection and difference between this sparse
        vector and the specified one.

        Please note that this operation is not commutative.  That is,
        given two sparse vectors A and B, venn(A) != venn(B) due to the
        potentially differing sets of indices per vector.

        :param other: A sparse vector to use.
        :return: A tuple whose first element is the intersection and the
        second the difference between two sparse vectors.
        """
        in_idx = np.in1d(self.indices, other.indices)
        diff_idx = np.in1d(self.indices,
                           np.setdiff1d(self.indices, self.indices[in_idx]))

        return SparseVector(self.data[in_idx], self.indices[in_idx],
                            self.size),\
               SparseVector(self.data[diff_idx], self.indices[diff_idx],
                            self.size)

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

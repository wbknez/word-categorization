"""
Contains classes and functions pertaining to sparse data structures.
"""
import numpy as np


class SparseMatrix:
    """
    Represents a matrix that only stores non-zero elements, making it memory
    efficient for managing large amounts of sparse data.

    Attributes:
        cols (np.array): The array of column indices.
        data (np.array): The array of non-zero elements.
        rows (np.array): The array of row indices.
        shape (tuple): The "true" dimensions.
    """

    def __init__(self, data, rows, cols, shape):
        self.cols = cols
        self.data = data
        self.rows = rows
        self.shape = shape

    def __getattr__(self, item):
        if item == "cols":
            return self.shape[1]
        elif item == "dtype":
            return self.data.dtype
        elif item == "rows":
            return self.shape[0]

    def __len__(self):
        """
        Returns the number of non-zero elements in this sparse matrix.

        Please note that the length is not the same as the size.

        :return: The number of non-zero elements.
        """
        return len(self.data)

    def get_data_column(self, index):
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

    def get_data_row(self, index):
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

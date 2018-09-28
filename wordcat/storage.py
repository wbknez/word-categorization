"""
Contains all classes and functions related to storing data for a learning
algorithm to use and operate upon.
"""
import numpy as np


class TrainingDatabase:
    """
    Represents a table of data that a learning algorithm may use to train
    itself and thereby learn about the hypothesis space in which it is
    expected to operate.

    Please note that this object is sparse and, since it is intended to keep
    memory low, has some severe restrictions in terms of format.  In
    particular:
      - All classes are expected to be 8-bit unsigned integers.
      - All counts are expected to be 16-bit unsigned integers.
      - All count indices are expected, but not required, to be 32-bit
      unsigned integers.
    
    Attributes:
    """

    def __init__(self, classes, counts):
        if classes.dtype != np.uint8:
            raise ValueError("Classes are expected to be 8-bit unsigned "
                             "integers.")
        if counts.dtype != np.uint16:
            raise ValueError("Counts are expected to be 16-bit unsigned "
                             "integers.")
        if counts.rows != len(classes):
            raise ValueError("The number of rows does not match the number of "
                             "classes.  The number of rows is: {} but the "
                             "number of classes is: "
                             "{}".format(counts.rows, len(classes)))

        self.classes = classes
        self.counts = counts

    def __getattr__(self, item):
        if item == "cols":
            return self.counts.cols
        elif item == "dtype":
            return self.counts.dtype
        elif item == "rows":
            return self.counts.rows

    def get_class_frequencies(self):
        """
        Computes and returns the frequencies of all unique classes in this
        training database.

        The returned value is actually a tuple of two items:
          1. An array of all unique classes found in this training database, and
          2. an array of their computed frequencies relative to one another.

        The formula for a single frequency is given as:
            f(C_i) = (number of classes of type C_i) / (total number of classes)

        :return: An tuple consisting of one array of unique classes and
        another of their frequencies.
        """
        class_counts = np.unique(self.classes, return_counts=True)
        return class_counts[0], np.divide(class_counts[1], len(self.classes))

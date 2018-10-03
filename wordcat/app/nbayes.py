"""
The main driver for the Word Categorization project using a naive Bayes
learning algorithm.

This project aims to use both naive Bayes and logistic regression algorithms to
explore document/text classification as a problem for applied machine learning.

This project is an assignment for CS 529, Fall 2018 with Dr. Estrada.
"""
import sys
from argparse import ArgumentParser

from wordcat.utils import DebugConsole


def parse_arguments():
    """
    Creates and configures the command line parser for this project with all
    necessary options and their data (arguments, descriptions) before parsing
    any options the user may have provided.

    :return: All parsed command line options.
    """
    parser = ArgumentParser()

    parser.add_argument("-b", "--beta",
                        help="Laplace smoothing coefficient",
                        type=float, default=(1/61188))
    parser.add_argument("-c", "--color",
                        help="enable colored console output",
                        action="store_true",
                        default=False)
    parser.add_argument("-p", "--parallel",
                        help="number of parallel processes",
                        type=int, default=4)
    parser.add_argument("-v", "--verbose",
                        help="enable verbose console output",
                        action="store_true",
                        default=False)

    subparsers = parser.add_subparsers(dest="subparsers")

    tester = subparsers.add_parser("test")

    tester.add_argument("dir", type=str,
                        help="path to data directory in CSV or PKL format")
    tester.add_argument("output", type=str,
                        help="path for output file")

    validator = subparsers.add_parser("validate")

    validator.add_argument("dir", type=str,
                           help="path to data directory in CSV or PKL format")
    validator.add_argument("folds", type=int,
                           help="number of times to fold data for validation")

    return parser.parse_args(sys.argv)


def test(tdb, dbgc, args):
    """

    :param tdb:
    :param dbgc:
    :param args:
    """
    pass


def validate(tdb, dbgc, args):
    """

    :param tdb:
    :param dbgc:
    :param args:
    """
    pass


def main():
    """
    The application entry point.

    :return: An exit code.
    """
    args = parse_arguments()
    dbgc = DebugConsole(args.color, args.verbose)


if __name__ == "__main__":
    sys.exit(main())

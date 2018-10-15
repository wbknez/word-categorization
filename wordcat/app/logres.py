"""
The main driver for the Word Categorization project using a logistic regression
learning algorithm.

This project aims to use both naive Bayes and logistic regression algorithms to
explore document/text classification as a problem for applied machine learning.

This project is an assignment for CS 529, Fall 2018 with Dr. Estrada.
"""
import os
import sys
from argparse import ArgumentParser
from multiprocessing import Pool

from wordcat.algorithm import LogisticRegressionLearningAlgorithm
from wordcat.app import test, validate
from wordcat.io import PickleIO
from wordcat.utils import DebugConsole


def parse_arguments():
    """
    Creates and configures the command line parser for this project with all
    necessary options and their data (arguments, descriptions) before parsing
    any options the user may have provided.

    :return: All parsed command line options.
    """
    parser = ArgumentParser()

    parser.add_argument("-c", "--color",
                        help="enable colored console output",
                        action="store_true",
                        default=False)
    parser.add_argument("-e", "--eta",
                        help="the learning rate",
                        type=float, default=0.01)
    parser.add_argument("-l", "--lambda_",
                        help="the penalty rate",
                        type=float, default=0.01)
    parser.add_argument("-p", "--parallel",
                        help="number of parallel processes",
                        type=int, default=4)
    parser.add_argument("-s", "--steps",
                        help="maximum number of weight refinements",
                        type=int, default=1000)
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

    return parser.parse_args()


def main():
    """
    The application entry point.

    :return: An exit code.
    """
    args = parse_arguments()
    dbgc = DebugConsole(args.color, args.verbose)

    if not os.path.exists(args.dir) or not os.path.isdir(args.dir):
        dbgc.fatal("Input directory does not exist: {}.", args.dir)
        dbgc.fatal("Cannot read data - please ensure the path is correct and "
                   "try again.")
        sys.exit(1)

    labels_path = os.path.join(args.dir, "newsgrouplabels.pkl")
    vocab_path = os.path.join(args.dir, "vocabulary.pkl")

    with Pool(processes=args.parallel) as pool:
        with open(labels_path, "rb") as labels_stream, \
                open(vocab_path, "rb") as vocab_stream:
            labels = PickleIO.read_labels(labels_stream)
            vocab = PickleIO.read_vocabulary(vocab_stream)

        learner = LogisticRegressionLearningAlgorithm(labels, vocab)

        if args.subparsers == "test":
            test(learner, pool, args, dbgc)
        else:
            validate(learner, pool, args, dbgc)


if __name__ == "__main__":
    sys.exit(main())

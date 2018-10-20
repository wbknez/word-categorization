"""
The main driver for the Word Categorization project using a naive Bayes
learning algorithm.

This project aims to use both naive Bayes and logistic regression algorithms to
explore document/text classification as a problem for applied machine learning.

This project is an assignment for CS 529, Fall 2018 with Dr. Estrada.
"""
import os
import sys
from argparse import ArgumentParser
from multiprocessing import Pool

from wordcat.algorithm import NaiveBayesLearningAlgorithm
from wordcat.app import test, validate
from wordcat.io import CsvIO, PickleIO
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
    tester.add_argument("-r", "--rank-words",
                        help="rank words according to relative importance",
                        action="store_true",
                        default=False)
    tester.add_argument("--rank-count", type=int,
                        help="how many words to rank",
                        default=100)
    tester.add_argument("--rank-output", type=str,
                        help="CSV file to write ranking results",
                        default="rankings.csv")

    validator = subparsers.add_parser("validate")

    validator.add_argument("dir", type=str,
                           help="path to data directory in CSV or PKL format")
    validator.add_argument("folds", type=int,
                           help="number of times to fold data for validation")
    validator.add_argument("--confusion-output", type=str,
                           help="path to write confusion matrix CSV",
                           default="confusion.csv")

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

        learner = NaiveBayesLearningAlgorithm(labels, vocab)

        if args.subparsers == "test":
            test(learner, pool, args, dbgc)

            if args.rank_words:
                dbgc.info("Outputting word rankings...")
                dbgc.info("Writing ranks of top {} words to {}.".format(
                    args.rank_count, args.rank_output
                ))
                with open(args.rank_output, "w+") as stream:
                    CsvIO.write_rankings(stream, ["word", "rank"],
                                         learner.rankings)
                dbgc.success("Finished writing word rankings.")
        else:
            validate(learner, pool, args, dbgc)


if __name__ == "__main__":
    sys.exit(main())

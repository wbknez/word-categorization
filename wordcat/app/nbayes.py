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

    validator = subparsers.add_parser("validate")

    validator.add_argument("dir", type=str,
                           help="path to data directory in CSV or PKL format")
    validator.add_argument("folds", type=int,
                           help="number of times to fold data for validation")

    return parser.parse_args(sys.argv)


def test(learner, pool, args, dbgc):
    """
    Tests the specified learning algorithm after training it on a user-provided
    data set and writes the resulting predictions to a user-chosen output
    file in CSV format.

    :param learner: The learning algorithm to use.
    :param pool: The processing pool to use.
    :param args: The user-chosen parameters to use.
    :param dbgc: The debug console to use.
    """
    test_path = os.path.join(args.dir, "testing.pkl")
    train_path = os.path.join(args.dir, "training.pkl")

    dbgc.info("Reading training data.")
    with open(train_path, "rb") as stream:
        tdb = PickleIO.read_database(stream)

    dbgc.info("Reading testing data.")
    with open(test_path, "rb") as stream:
        ts = PickleIO.read_set(stream)
    dbgc.success("Configuration complete.")

    dbgc.info("Beginning to train...")
    learner.train(pool, tdb, args, dbgc)
    dbgc.success("Training complete.")

    dbgc.info("Beginning to test...")
    ids, classes = learner.test(ts, pool, dbgc)
    dbgc.success("Testing complete.")

    dbgc.info("Outputting predictions...")
    dbgc.info("Writing results to: {}.", args.output)
    with open(args.output, "w+") as stream:
        CsvIO.write_predictions(stream, ["id", "class"], ids, classes)
    dbgc.success("Finished writing results.")


def validate(learner, pool, args, dbgc):
    """
    Validates the specified learning algorithm after training it on a
    user-provided data set that has been stripped by a user-chosen amount of
    times for validation and reports the success rate and number of both
    correct and incorrect predictions.

    :param learner: The learning algorithm to use.
    :param pool: The processing pool to use.
    :param args: The user-chosen parameters to use.
    :param dbgc: The debug console to use.
    """
    pass


def main():
    """
    The application entry point.

    :return: An exit code.
    """
    if sys.argv and "wordcat/app/nbayes.py" in sys.argv[0]:
        sys.argv = sys.argv[1:]

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
        else:
            validate(learner, pool, args, dbgc)


if __name__ == "__main__":
    sys.exit(main())

"""
A utility to optimize data files for the Word Categorization project.

This file is part of an assignment for CS 529, Fall 2018 with Dr. Estrada.
"""
import os
import shutil
import sys
from argparse import ArgumentParser
from collections import namedtuple
from multiprocessing.pool import Pool
from timeit import default_timer as timer

from wordcat.io import DataIO, PickleIO
from wordcat.utils import DebugConsole


DataSpecification = namedtuple("FileSpecification", [
    "description", "input", "output", "read_method", "write_method"
])
DataSpecification.__doc__ = """A collection of metadata about a single data 
file to convert."""


def parse_arguments():
    """
    Creates and configures the command line parser for this project with all
    necessary options and their data (arguments, descriptions) before parsing
    any options the user may have provided.

    :return: All parsed command line options.
    """
    parser = ArgumentParser()

    parser.add_argument("input", type=str,
                        help="input directory of CSV formatted data")
    parser.add_argument("output", type=str,
                        help="output directory for PKL formatted data")

    parser.add_argument("-c", "--color",
                        help="enable colored console output",
                        action="store_true",
                        default=False)
    parser.add_argument("-f", "--force",
                        help="overwrite existing pickled data",
                        action="store_true",
                        default=False)
    parser.add_argument("-p", "--parallel",
                        help="number of parallel processes",
                        type=int, default=4)
    parser.add_argument("-v", "--verbose",
                        help="enable verbose console output",
                        action="store_true",
                        default=False)

    return parser.parse_args(sys.argv)


def process_data(spec, args, pool, dbgc):
    """
    Converts the object defined by the specified data specification from a
    non-sparse form to a sparse or Python object form, making it easy and
    efficient to load for repeated use.

    :param spec: The data file specification to use.
    :param args: The collection of command line arguments, if any.
    :param pool: The processing pool to use.
    :param dbgc: The debug console to use.
    """
    source = os.path.join(args.input, "newsgrouplabels.txt")
    dest = os.path.join(args.output, "newsgrouplabels.pkl")

    dbgc.info("Processing {}...", spec.description)
    dbgc.info("Reading object from: {}.", source)
    obj = spec.read_method(pool, source)

    dbgc.info("Writing object to: {}.", dest)
    with open(dest, "wb+") as stream:
        spec.write_method(stream, obj)

    dbgc.success("Finished processing.")


def main():
    """
    The application entry point.

    :return: An exit code.
    """
    if sys.argv and "wordcat/app/dataprep.py" in sys.argv[0]:
        sys.argv = sys.argv[1:]

    args = parse_arguments()
    dbgc = DebugConsole(args.color, args.verbose)
    specifications = [
        DataSpecification("class labels", "newsgrouplabels.txt",
                          "newgrouplabels.pkl",
                          DataIO.read_labels, PickleIO.write_labels),
        DataSpecification("testing data", "testing.csv", "testing.pkl",
                          DataIO.read_set, PickleIO.write_set),
        DataSpecification("training data", "training.csv", "training.pkl",
                          DataIO.read_database, PickleIO.write_database),
        DataSpecification("vocabulary set", "vocabulary.txt", "vocabulary.pkl",
                          DataIO.read_vocabulary, PickleIO.write_vocabulary)
    ]

    if not os.path.exists(args.input) or not os.path.isdir(args.input):
        dbgc.fatal("Input directory does not exist: {}.", args.input)
        dbgc.fatal("There is nothing to process.")
        sys.exit(1)

    if os.path.exists(args.output):
        if not args.force:
            dbgc.fatal("Output directory already exists: {}.", args.output)
            dbgc.fatal("Please move or delete it and try again.")
            sys.exit(1)
        else:
            dbgc.info("Removing previous output directory: {}.", args.output)
            shutil.rmtree(args.output)

    start = timer()

    dbgc.info("Creating new output directory: {}.", args.output)
    os.mkdir(args.output)

    with Pool(processes=args.parallel) as pool:
        for spec in specifications:
            process_data(spec, args, pool, dbgc)

    end = timer()
    dbgc.complete("Processing complete for all after {:.2f} seconds.",
                  end - start)

if __name__ == "__main__":
    sys.exit(main())

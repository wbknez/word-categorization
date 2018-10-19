"""

"""
import os
from math import floor

from wordcat.io import CsvIO, PickleIO


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
    predictions = learner.test(pool, ts, dbgc)
    dbgc.success("Testing complete.")

    dbgc.info("Outputting predictions...")
    dbgc.info("Writing results to: {}.", args.output)
    with open(args.output, "w+") as stream:
        CsvIO.write_predictions(stream, ["id", "class"], predictions)
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
    train_path = os.path.join(args.dir, "training.pkl")

    dbgc.info("Reading training data.")
    with open(train_path, "rb") as stream:
        tdb = PickleIO.read_database(stream)

    dbgc.info("Calculating folds.")


    dbgc.info("Beginning k-fold cross validation.")
    pass

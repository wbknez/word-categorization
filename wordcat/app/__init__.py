"""

"""
import os

from wordcat.io import CsvIO, PickleIO
from wordcat.storage import ConfusionMatrix


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
    if args.folds <= 1:
        raise ValueError("Fold must be at least two or greater.")

    train_path = os.path.join(args.dir, "training.pkl")

    dbgc.info("Reading training data.")
    with open(train_path, "rb") as stream:
        tdb = PickleIO.read_database(stream)
    dbgc.success("Configuration complete.")

    dbgc.info("Shuffling database.")
    tdb.shuffle()
    dbgc.success("Shuffling complete.")

    dbgc.info("Creating confusion matrix.")
    cfm = ConfusionMatrix(learner.labels)

    dbgc.info("Calculating folds.")
    folds = tdb.create_k_folds(args.folds)

    dbgc.info("Beginning k-fold cross validation...")
    for index, fold in enumerate(folds):
        dbgc.info("Splitting at fold: {} of: {}.".format(index + 1, len(folds)))
        ntdb, ts, ans = tdb.split(fold)

        dbgc.info("Beginning to train...")
        learner.train(pool, ntdb, args, dbgc)
        dbgc.success("Training complete.")

        dbgc.info("Beginning to validate...")
        vs = learner.validate(pool, ts, ans, dbgc)
        dbgc.success("Validation complete.")

        dbgc.info("Updating confusion matrix...")
        cfm.update(vs)
        dbgc.success("Confusion matrix updated.")

    dbgc.info("Outputting confusion matrix...")
    dbgc.info("Writing matrix to: {}.".format(args.confusion_output))
    with open(args.confusion_output, "w+") as stream:
        CsvIO.write_confusion(stream, [str(i) for i in range(1, 21)], cfm)
    dbgc.success("Finished writing confusion matrix.")

    dbgc.info("Computing accuracy...")
    total = sum(c for count in cfm.counts.values() for c in count.values())
    correct = sum([cfm.counts[i][i] for i in range(1, 21)])
    dbgc.info("Accuracy is: {:.2f}.".format((correct / total) * 100))
    dbgc.success("Finished computing accuracy.")

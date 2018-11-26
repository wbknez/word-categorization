Naive Bayes and Logistic Regression
=========================

Overview
--------
This project aims to implement two learning algorithms - naive Bayes and
logistic regression - in order to explore their effectiveness and performance on
a simple text classification problem.

This project was completed for UNM CS 529, Fall 2018.

Features
--------
The following are notable features implemented in this project:

 * Parallelized multinomial naive Bayes classifier.
 * Multinomial logistic regression classifier.
 * A simple data pipeline using Python's `pickle` library.
 * Sparse vector and matrix library using a set-based numeric space.
 * Term frequency inverse document frequency (TF-IDF) for finding and penalizing
   stop words.
 * Confusion matrix creation and output to CSV.
 * K-fold cross validation and accuracy reporting.

Sample Output
-------------
![Naive Bayes Confusion Matrix](images/bayes_ccfm.png)

Confusion matrix for a sample naive Bayes run.

Dependencies
------------
This project was written in Python 3 and requires the following libraries:

 * [Nose2](https://github.com/nose-devs/nose2) - a unit testing framework.
 * [NumPy](http://www.numpy.org/) - a library for numeric computing in Python.

Building
--------
To build this project, first install the dependencies with either:

```make deps```

or issuing the following command directly:

````python3 -m pip install --user -r requirements.txt````

Running
-------
This project consists of three different programs:
  1. `nbayes.sh` - Runs a naive Bayes classifier over a data set.
  2. `logres.sh` - Runs a logistic regression classifier over a data set.
  3. `sparsify.sh` - Prepares a directory of data for use.

First, before choosing a classifier, the data should be prepared for use.  To do
so, simply call the `sparsify` utility and pass it the paths you would like it
to utilize.  To do so, execute the following:

```./sparsify.sh [OPTIONS] [INPUT] [OUTPUT]```

where:

 * `INPUT` is the path to a Kaggle data directory.
 * `OUTPUT` is the path to a non-existent output directory.

Once the data has been formatted, they may be used with a learning algorithm to
perform text classification.  The two classifiers in this project are: `nbayes`
and `logres`.

Each program has two subcommands: `test` and `validate`.

To train and test a classifier for use with Kaggle, execute the following:

```./nbayes.sh [OPTIONS] test [INPUT] [OUTPUT]```

or:

```./logres.sh [OPTIONS] test [INPUT] [OUTPUT]```

where:

 * `INPUT` is a path to a directory with pre-processed data files.
 * `OUTPUT` is a path to a local file where the program should write any 
 predictive results for submission.

To train a learner and have it validate itself on a single training database,
 execute the following:
 
 ```./nbayes.sh [OPTIONS] validate [INPUT] [FOLDS]```
 
 or:

 ```./logres.sh [OPTIONS] validate [INPUT] [FOLDS]```

where:

 * `INPUT` is a path to a directory with pre-processed data files.
 * `FOLDS` is the number of sub-intervals of the training data to use for
   testing and subsequent validation.

Command Line Options
--------------------
The following command line options are available to all three programs:

 * `-c`, `--color` - Enable ANSI sequences to colorize program output.
 * `-p`, `--parallel` - The number of additional processes to use to boost
   performance.  This is `4` by default.
 * `-v`, `--verbose` - Enable verbose output.  This is disabled by default.

The following command line options are only available to `sparsify`:

 * `-f`, `--force` - Force the utility to overwrite an existing output
   directory.  This is disabled by default.

The following command line options are available to both `nbayes` and `logres`:
 * `--confusion-output` - The path to a CSV file to which the confusion matrix
   for a single run will be written.  This is `confusion.csv` by default.
 * `--make-confusion` - Whether or not to write the confusion matrix to CSV.
   This is disabled by default.

The following command line options are only available to `nbayes`:
 * `-b`, `--beta` - The learning rate.  This is `1/61188` by default.
 * `-r`, `--rank-words` - Whether or not to compute the TF-IDF importance
   ranking for each word in a vocabulary.
 * `--rank-count` - How many words to count and output the rankings of.
 * `--rank-output` - The path to the CSV to which to write the word rankings.

The following command line options are only available to `logres`:
 * `-e`, `--eta` - The learning rate.  By default this is `0.01`.
 * `-l`, `--lambda_` - The penalty term.  By default this is `0.01`.
 * `-s`, `--steps` - The number of iterations to execute before stopping.  By
   default this is `250`.

Unit Tests
----------
Unit tests may be run with either the following command:

```make test```

or:

```python3 -m nose2 tests```

License
-------
This project is released under the 
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license as 
specified in [License.txt](License.txt).

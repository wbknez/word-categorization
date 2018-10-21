#!/usr/bin/env sh

# sparsify.sh
#
# A utility to convert Kaggle data sets to pre-formatted Python objects to be
# consumed by learning algorithms.
python3 -m wordcat.app.sparsify "$@"

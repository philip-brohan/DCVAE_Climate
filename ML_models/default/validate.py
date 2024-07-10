#!/usr/bin/env python

# Plot a validation figure for the autoencoder.

# For all outputs:
#  1) Target field
#  2) Autoencoder output
#  3) scatter plot

import os

# Supress TensorFlow moaning about cuda - we don't need a GPU for this
# Also the warning message confuses people.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from ML_models.default.makeDataset import getDataset
from ML_models.default.autoencoderModel import getModel
from ML_models.default.gmUtils import plotValidationField

from specify import specification

specification["strategy"] = (
    tf.distribute.get_strategy()
)  # No distribution for simple validation


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=500)
parser.add_argument("--year", help="Test year", type=int, required=False, default=None)
parser.add_argument(
    "--month", help="Test month", type=int, required=False, default=None
)
parser.add_argument(
    "--training",
    help="Use training data (not test)",
    default=False,
    action="store_true",
)
args = parser.parse_args()

purpose = "Test"
if args.training:
    purpose = "Train"
# Go through data and get the desired month
dataset = (
    getDataset(specification, purpose=purpose)
    .shuffle(specification["shuffleBufferSize"])
    .batch(1)
)
input = None
year = None
month = None
for batch in dataset:
    dateStr = batch[0][0].numpy().decode("utf-8")
    year = int(dateStr[:4])
    month = int(dateStr[5:7])
    if (args.month is None or month == args.month) and (
        args.year is None or year == args.year
    ):
        input = batch
        break

if input is None:
    raise Exception("Month %04d-%02d not in %s dataset" % (year, month, purpose))

autoencoder = getModel(specification, args.epoch)

# Get autoencoded tensors
output = autoencoder.call(input, training=False)

# Make the plot
plotValidationField(specification, input, output, year, month, "comparison.webp")

#!/usr/bin/env python

# Plot a validation figure for the autoencoder.

# For all outputs:
#  1) Target field
#  2) Autoencoder output
#  3) scatter plot

import tensorflow as tf


from ML_models.mk1.makeDataset import getDataset
from ML_models.mk1.autoencoderModel import getModel

from ML_models.mk1.gmUtils import plotValidationField

from specify import specification

specification["strategy"] = (
    tf.distribute.get_strategy()
)  # No distribution for simple validation

# I don't need all the messages about a missing font (on Isambard)
import logging

logging.getLogger("matplotlib.font_manager").disabled = True

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
    dateStr = tf.strings.split(batch[0][0][0], sep="/")[-1].numpy()
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

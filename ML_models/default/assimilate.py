#!/usr/bin/env python

# Find a point in latent space that maximises the fit to some given input fields,
#  and plot the fitted state.

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ML_models.default.makeDataset import getDataset
from ML_models.default.autoencoderModel import getModel

from ML_models.default.gmUtils import plotValidationField

from specify import specification

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=250)
parser.add_argument(
    "--year", help="Year to fit to", type=int, required=False, default=1969
)
parser.add_argument(
    "--month", help="Month to fit to", type=int, required=False, default=3
)
fitted = []
for field in specification["outputNames"]:
    parser.add_argument(
        "--%s" % field,
        help="Fit to %s?" % field,
        dest=fitted[field],
        default=False,
        action="store_true",
    )
masked = []
for field in specification["outputNames"]:
    parser.add_argument(
        "--%s_mask" % field,
        help="Mask for fit to %s?" % field,
        type=str,
        dest=masked[field],
        default=None,
    )
parser.add_argument(
    "--iter",
    help="No. of iterations",
    type=int,
    required=False,
    default=100,
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

# We are using the model in inference mode - (does this have any effect?)
autoencoder.trainable = False

latent = tf.Variable(tf.random.normal(shape=(1, autoencoder.latent_dim)))
if specification["outputTensors"] is not None:
    target = tf.constant(input[2][0], dtype=tf.float32)
else:
    target = tf.constant(input[1][0], dtype=tf.float32)


def decodeFit():
    result = 0.0
    generated = autoencoder.generate(latent, training=False)
    for field in specification["outputNames"]:
        if fitted[field]:
            mask = masked[field]
            if mask is not None:
                mask = tf.constant(mask, dtype=tf.float32)
                result = result + tf.reduce_mean(
                    tf.keras.metrics.mean_squared_error(
                        tf.boolean_mask(generated[:, :, :, field], mask),
                        tf.boolean_mask(target[:, :, :, field], mask),
                    )
                )
            else:
                result = result + tf.reduce_mean(
                    tf.keras.metrics.mean_squared_error(
                        generated[:, :, :, field], target[:, :, :, field]
                    )
                )
    return result


# If anything to assimilate, search for the latent space point that minimises the loss
if any(fitted.values()):
    loss = tfp.math.minimize(
        decodeFit,
        trainable_variables=[latent],
        num_steps=args.iter,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
    )

# Output is the generated value from the fitted latent space point
generated = autoencoder.generate(latent, training=False)

# Make the plot - same as for validation script
plotValidationField(specification, input, generated, year, month, "assimilated.webp")

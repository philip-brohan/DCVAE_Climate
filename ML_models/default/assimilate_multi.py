#!/usr/bin/env python

# Plot fit statistics for all the test cases

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
from statistics import mean

from specify import specification


from ML_models.default.makeDataset import getDataset
from ML_models.default.autoencoderModel import getModel
from ML_models.default.gmUtils import (
    computeScalarStats,
    plotScalarStats,
)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=500)
parser.add_argument(
    "--startyear", help="First year to plot", type=int, required=False, default=None
)
parser.add_argument(
    "--endyear", help="Last year to plot", type=int, required=False, default=None
)
parser.add_argument(
    "--xpoint",
    help="Extract data at this x point",
    type=int,
    required=False,
    default=None,
)
parser.add_argument(
    "--ypoint",
    help="Extract data at this y point",
    type=int,
    required=False,
    default=None,
)
parser.add_argument(
    "--training",
    help="Use training months (not test months)",
    dest="training",
    default=False,
    action="store_true",
)
for field in specification["outputNames"]:
    parser.add_argument(
        "--%s" % field,
        help="Fit to %s?" % field,
        default=False,
        action="store_true",
    )
for field in specification["outputNames"]:
    parser.add_argument(
        "--%s_mask" % field,
        help="Mask for fit to %s?" % field,
        type=str,
        default=None,
    )
parser.add_argument(
    "--iter",
    help="No. of iterations",
    type=int,
    required=False,
    default=100,
)
args = parser.parse_args()
args_dict = vars(args)
# Load the masks, if specified
fitted = {}
masked = {}
for field in specification["outputNames"]:
    masked[field] = None
    if args_dict["%s_mask" % field] is not None:
        masked[field] = np.load(args_dict["%s_mask" % field])
    fitted[field] = args_dict[field]

# Set up the test data
purpose = "Test"
if args.training:
    purpose = "Train"
testData = getDataset(specification, purpose=purpose)
testData = testData.batch(1)


# Load the trained model
autoencoder = getModel(specification, args.epoch)


def decodeFit():
    result = 0.0
    generated = autoencoder.generate(latent, training=False)
    for field in specification["outputNames"]:
        if fitted[field]:
            field_idx = specification["outputNames"].index(field)
            mask = masked[field]
            if mask is not None:
                mask = tf.constant(mask, dtype=tf.float32)
                result = result + tf.reduce_mean(
                    tf.keras.metrics.mean_squared_error(
                        tf.boolean_mask(generated[0, :, :, field_idx], mask),
                        tf.boolean_mask(target[:, :, field_idx], mask),
                    )
                )
            else:
                result = result + tf.reduce_mean(
                    tf.keras.metrics.mean_squared_error(
                        generated[:, :, :, field_idx], target[:, :, field_idx]
                    )
                )
    return result


# Go through the data and get the scalar stat for each test month
all_stats = {}
all_stats["dtp"] = []
all_stats["target"] = {}
all_stats["generated"] = {}
for case in testData:
    if specification["outputTensors"] is not None:
        target = tf.constant(case[2][0], dtype=tf.float32)
    else:
        target = tf.constant(case[1][0], dtype=tf.float32)
    latent = tf.Variable(autoencoder.makeLatent())
    if any(fitted.values()):
        loss = tfp.math.minimize(
            decodeFit,
            trainable_variables=[latent],
            num_steps=args.iter,
            optimizer=tf.optimizers.Adam(learning_rate=0.1),
        )
    generated = autoencoder.generate(latent, training=False)
    stats = computeScalarStats(
        specification,
        case,
        generated,
    )
    all_stats["dtp"].append(stats["dtp"])
    for key in stats["target"].keys():
        if key in all_stats["target"]:
            all_stats["target"][key].append(stats["target"][key])
            all_stats["generated"][key].append(stats["generated"][key])
        else:
            all_stats["target"][key] = [stats["target"][key]]
            all_stats["generated"][key] = [stats["generated"][key]]

# Make the plot
plotScalarStats(all_stats, specification, fileName="assimilate_multi.webp")

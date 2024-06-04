#!/usr/bin/env python

# Find a point in latent space that maximises the fit to some given input fields,
#  and plot the fitted state.
    #
    # assimilate.py --epoch 500 --year 1990 --day 786

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

sys.path.append('/home/h03/hadsx/extremes/ML/pb1/DCVAE_Climate_sjb1/cpm1')
from ML_models.mk1.makeDataset import getDataset
from ML_models.mk1.autoencoderModel import getModel

from ML_models.mk1.gmUtils import plotValidationField

from specify import specification

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=500)
parser.add_argument("--year", help="Test year", type=int, required=False, default=1990)
parser.add_argument(
    "--day", help="Test day", type=int, required=False, default=786
)
for field in specification["outputNames"]:
    parser.add_argument(
        "--%s" % field,
        help="Fit to %s?" % field,
        default=False,
        action="store_true",
    )
# for field in specification["outputNames"]:
#     parser.add_argument(
#         "--%s_mask" % field,
#         help="Mask for fit to %s?" % field,
#         type=str,
#         default=False,  # None,
#     )
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
args_dict = vars(args)
# Load the masks, if specified
fitted = {}
# masked = {}
for field in specification["outputNames"]:
    # masked[field] = None
    # if args_dict["%s_mask" % field] is not None:
    #     masked[field] = np.load(args_dict["%s_mask" % field])
    fitted[field] = args_dict[field]

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
    # print(dateStr)
    # year = int(dateStr[:4])
    # month = int(dateStr[5:7])
    year = int(dateStr[10:14]) # int(fN[:4])
    # month = int(fN[5:7])
    idot    = str(dateStr).find('.')
    day     = int(dateStr[15:(idot-2)]) # int(fN[5:7])
    if (day == args.day) and (year == args.year):
        input = batch
        break
    # input = batch

if input is None:
    raise Exception("%04d-%02d not in %s dataset" % (year, day, purpose))

autoencoder = getModel(specification, args.epoch)

# We are using the model in inference mode - (does this have any effect?)
autoencoder.trainable = False

latent = tf.Variable(autoencoder.makeLatent())
if specification["outputTensors"] is not None:
    target = tf.constant(input[2][0], dtype=tf.float32)
else:
    target = tf.constant(input[1][0], dtype=tf.float32)


def decodeFit():
    result = 0.0
    generated = autoencoder.generate(latent, training=False)
    for field in specification["outputNames"]:
        if fitted[field]:
            field_idx = specification["outputNames"].index(field)
            # mask = masked[field]
            # mask = None
            # if mask is not None:
            #     mask = tf.constant(mask, dtype=tf.float32)
            #     result = result + tf.reduce_mean(
            #         tf.keras.metrics.mean_squared_error(
            #             tf.boolean_mask(generated[0, :, :, field_idx], mask),
            #             tf.boolean_mask(target[:, :, field_idx], mask),
            #         )
            #     )
            # else:
            result = result + tf.reduce_mean(
                tf.keras.metrics.mean_squared_error(
                    generated[:, :, :, field_idx], target[:, :, field_idx]
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
title0="%s e%d %04d-%3d" % (specification["modelName"], args.epoch, year, day)
savefile0="ass_%s_e%d_%04d-%3d.webp" % (specification["modelName"], args.epoch, year, day)
plotValidationField(specification, input, generated, year, day, savefile0, title0)

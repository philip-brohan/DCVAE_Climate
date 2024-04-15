#!/usr/bin/env python

# Make normalized tensors

import os
import sys
import argparse
import tensorflow as tf
from normalize.ERA5.makeDataset import getDataset
from normalize.ERA5.normalize import match_normal, load_fitted

sDir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--variable",
    help="Variable name",
    type=str,
    required=True,
)
args = parser.parse_args()

opdir = "%s/DCVAE-Climate/normalized_datasets/ERA5/%s/" % (
    os.getenv("SCRATCH"),
    args.variable,
)
if not os.path.isdir(opdir):
    os.makedirs(opdir)


def is_done(year, month):
    fn = "%s/%04d-%02d.tfd" % (
        opdir,
        year,
        month,
    )
    if os.path.exists(fn):
        return True
    return False


# Load the pre-calculated normalisation parameters
fitted = []
for month in range(1, 13):
    cubes = load_fitted(month, variable=args.variable)
    fitted.append([cubes[0].data, cubes[1].data, cubes[2].data])


# Go through raw dataset  and make normalized tensors
trainingData = getDataset(
    args.variable,
    cache=False,
    blur=1.0e-9,
).batch(1)


for batch in trainingData:
    year = int(batch[1].numpy()[0][0:4])
    month = int(batch[1].numpy()[0][5:7])
    if is_done(year, month):
        continue

    # normalize
    raw = batch[0].numpy().squeeze()
    normalized = match_normal(raw, fitted[month - 1])
    ict = tf.convert_to_tensor(normalized, tf.float32)
    tf.debugging.check_numerics(ict, "Bad data %04d-%02d" % (year, month))

    # Write to file
    opfile = ("%s/%04d-%02d.tfd") % (
        opdir,
        year,
        month,
    )
    sict = tf.io.serialize_tensor(ict)
    tf.io.write_file(opfile, sict)

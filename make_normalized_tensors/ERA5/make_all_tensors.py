#!/usr/bin/env python

# Make normalized tensors

import os
import sys
import argparse
import tensorflow as tf
from normalize.ERA5.makeDataset import getDataset
from normalize.ERA5.normalize import match_normal, load_fitted
import tensorstore as ts
from shutil import rmtree
import zarr

sDir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--variable",
    help="Variable name",
    type=str,
    required=True,
)
args = parser.parse_args()

# Get the date range from the input zarr array
fn = "%s/DCVAE-Climate/raw_datasets/ERA5/%s_zarr" % (
    os.getenv("SCRATCH"),
    args.variable,
)
input_zarr = zarr.open(fn, mode="r")


def date_to_index(year, month):
    return (year - input_zarr.attrs["FirstYear"]) * 12 + month - 1


# Create the output zarr array
fn = "%s/DCVAE-Climate/normalized_datasets/ERA5/%s_zarr" % (
    os.getenv("SCRATCH"),
    args.variable,
)
# Delete any previous version
if os.path.exists(fn):
    rmtree(fn)

normalized_zarr = ts.open(
    {
        "driver": "zarr",
        "kvstore": "file://" + fn,
    },
    dtype=ts.float32,
    chunk_layout=ts.ChunkLayout(chunk_shape=[721, 1440, 1]),
    create=True,
    shape=[
        721,
        1440,
        date_to_index(input_zarr.attrs["LastYear"], 12) + 1,
    ],
).result()
normalized_zarr.attrs["FirstYear"] = input_zarr.attrs["FirstYear"]
normalized_zarr.attrs["LastYear"] = input_zarr.attrs["LastYear"]

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

    # normalize
    raw = batch[0].numpy().squeeze()
    normalized = match_normal(raw, fitted[month - 1])
    ict = tf.convert_to_tensor(normalized, tf.float32)
    tf.debugging.check_numerics(ict, "Bad data %04d-%02d" % (year, month))

    didx = date_to_index(year, month)
    op = normalized_zarr[:, :, didx].write(ict)

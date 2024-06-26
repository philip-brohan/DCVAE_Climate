#!/usr/bin/env python

# Read in monthly variable from ERA5 - regrid to model resolution
# Convert into a TensorFlow tensor.
# Serialise and store on $SCRATCH.

import os
import sys

# Supress TensorFlow moaning about cuda - we don't need a GPU for this
# Also the warning message confuses people.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorstore as ts

import dask

# Going to do external parallelism - run this on one core
tf.config.threading.set_inter_op_parallelism_threads(1)
dask.config.set(scheduler="single-threaded")

from tensor_utils import load_raw, raw_to_tensor, date_to_index

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--month", help="Integer month", type=int, required=True)
parser.add_argument("--variable", help="Variable name", type=str, required=True)
args = parser.parse_args()

fn = "%s/DCVAE-Climate/raw_datasets/ERA5/%s_zarr" % (
    os.getenv("SCRATCH"),
    args.variable,
)

dataset = ts.open(
    {
        "driver": "zarr",
        "kvstore": "file://" + fn,
    }
).result()

# Load and standardise data
qd = load_raw(args.year, args.month, variable=args.variable)
ict = raw_to_tensor(qd)

# Write to file
didx = date_to_index(args.year, args.month)
op = dataset[:, :, didx].write(ict)

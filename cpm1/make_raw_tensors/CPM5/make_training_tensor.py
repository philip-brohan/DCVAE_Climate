#!/usr/bin/env python

# Read in daily variable from CPM5 - regrid to model resolution
# Convert into a TensorFlow tensor.
# Serialise and store on $SCRATCH.

import os
import sys
import tensorflow as tf
import dask

# Going to do external parallelism - run this on one core
tf.config.threading.set_inter_op_parallelism_threads(1)
dask.config.set(scheduler="single-threaded")

from tensor_utils import load_raw, raw_to_tensor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--member", help="Year", type=int, required=True)
parser.add_argument("--variable", help="Variable name", type=str, required=True)
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument("--day", help="Integer day", type=int, required=True)
parser.add_argument(
    "--opfile", help="tf data file name", default=None, type=str, required=False
)
args = parser.parse_args()
if args.opfile is None:
    args.opfile = ("/scratch/hadsx/cpm/5km/daily/1day/%s/raw_tensors/member_%02d_%04d_%d.tfd") % (
        args.variable,
        args.member,
        args.year,
        args.day,
    )


if not os.path.isdir(os.path.dirname(args.opfile)):
    os.makedirs(os.path.dirname(args.opfile))

# Load and standardise data
qd = load_raw(member=args.member, variable=args.variable, year=args.year, day=args.day)
ict = raw_to_tensor(qd)

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile, sict)

#!/usr/bin/env python

# Make normalized tensors

import os
import sys
import argparse
import tensorflow as tf
from normalize.CPM5.makeDataset import getDataset
from normalize.CPM5.normalize import match_normal, load_fitted


# module load scitools
# python -i ./make_all_tensors.py  --variable=tas

# exec(open('./make_1day_files.py').read())

sDir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--variable",
    help="Variable name",
    type=str,
    required=True,
)
args = parser.parse_args()

opdir = "/scratch/hadsx/cpm/5km/daily/1day/%s/norm_tensors" % (
    args.variable,
)
if not os.path.isdir(opdir):
    os.makedirs(opdir)

def is_done(opdir, member, year, day):
    fn = ("%s/member_%02d_%04d_%d.tfd") % (
        opdir,
        member,
        year,
        day,
    )
    if os.path.exists(fn):
        return True
    return False


# Load the pre-calculated normalisation parameters
fitted = []
for month in range(6, 7):
    cubes = load_fitted(month, variable=args.variable)
    fitted.append([cubes[0].data, cubes[1].data, cubes[2].data])


# Go through raw dataset  and make normalized tensors
trainingData = getDataset(
    args.variable,
    cache=False,
    blur=1.0e-9,
).batch(1)


for batch in trainingData:
    member  = int(batch[1].numpy()[0][7:9])
    year    = int(batch[1].numpy()[0][10:14])
    fstring = str(batch[1].numpy()[0])
    idot    = fstring.find('.')
    day     = int(batch[1].numpy()[0][15:(idot-2)])
    if is_done(opdir, member, year, day):
        continue

    # normalize
    raw = batch[0].numpy().squeeze()
    # normalized = match_normal(raw, fitted[month - 1])
    normalized = match_normal(raw, fitted[0])
    ict = tf.convert_to_tensor(normalized, tf.float32)
    tf.debugging.check_numerics(ict, "Bad data month %02d" % (month))

    # Write to file
    opfile = ("%s/member_%02d_%04d_%d.tfd") % (
        opdir,
        member,
        year,
        day,
    )
    sict = tf.io.serialize_tensor(ict)
    tf.io.write_file(opfile, sict)

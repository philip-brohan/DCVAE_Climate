#!/usr/bin/env python

# Make raw data tensors for normalization

import os
import argparse
import tensorstore as ts

from tensor_utils import date_to_index, FirstYear, LastYear

sDir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--variable",
    help="Variable name",
    type=str,
    required=True,
)
args = parser.parse_args()

# Create the output zarr array if it doesn't exist
fn = "%s/DCVAE-Climate/raw_datasets/ERA5/%s_zarr" % (
    os.getenv("SCRATCH"),
    args.variable,
)
if not os.path.exists(fn):
    dataset = ts.open(
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
            date_to_index(LastYear, 12) + 1,
        ],
    ).result()


count = 0
for year in range(FirstYear, LastYear + 1):
    for month in range(1, 13):
        cmd = "%s/make_training_tensor.py --year=%04d --month=%02d --variable=%s" % (
            sDir,
            year,
            month,
            args.variable,
        )
        print(cmd)

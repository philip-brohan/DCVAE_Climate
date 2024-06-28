#!/usr/bin/env python

# Make raw data tensors for normalization

import os
from shutil import rmtree
import argparse
import zarr
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
# Clear out last version
if os.path.exists(fn):
    rmtree(fn)

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

# Add date range to array as metadata
# TensorStore doesn't support metadata, so use the underlying zarr array
zarr_ds = zarr.open(fn, mode="r+")
zarr_ds.attrs["FirstYear"] = FirstYear
zarr_ds.attrs["LastYear"] = LastYear

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

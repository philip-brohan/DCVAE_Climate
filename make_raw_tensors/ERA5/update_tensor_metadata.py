#!/usr/bin/env python

# Update the raw tensor zarr array with metadata giving dates and indices for each field present

import os
import argparse
import zarr
import numpy as np

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

# Find the raw_tensor zarr array
fn = "%s/DCVAE-Climate/raw_datasets/ERA5/%s_zarr" % (
    os.getenv("SCRATCH"),
    args.variable,
)

# Add date range to array as metadata
zarr_ds = zarr.open(fn, mode="r+")

AvailableMonths = {}
for year in range(FirstYear, LastYear + 1):
    for month in range(1, 13):
        idx = date_to_index(year, month)
        slice = zarr_ds[:, :, idx]
        if not np.all(np.isnan(slice)):
            AvailableMonths["%d-%02d" % (year, month)] = idx

zarr_ds.attrs["AvailableMonths"] = AvailableMonths

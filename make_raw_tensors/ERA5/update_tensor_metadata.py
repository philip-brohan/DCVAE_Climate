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
start = "%04d-%02d" % (LastYear, 12)
end = "%04d-%02d" % (FirstYear, 1)
for year in range(FirstYear, LastYear + 1):
    for month in range(1, 13):
        dte = "%d-%02d" % (year, month)
        idx = date_to_index(year, month)
        slice = zarr_ds[:, :, idx]
        if not np.all(np.isnan(slice)):
            AvailableMonths["%d-%02d" % (year, month)] = idx
            if dte < start:
                start = dte
            if dte > end:
                end = dte

zarr_ds.attrs["AvailableMonths"] = AvailableMonths

missing = 0
for year in range(FirstYear, LastYear + 1):
    for month in range(1, 13):
        dte = "%d-%02d" % (year, month)
        if dte < start or dte > end:
            continue
        if dte not in AvailableMonths:
            missing += 1

print(args.variable)
print("Start date:", start)
print("End date:", end)
print("Missing months:", missing)
print("Total months:", len(AvailableMonths))
print("\n")

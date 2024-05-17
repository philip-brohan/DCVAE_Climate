#!/usr/bin/env python

# Retrieve CPM5 daily averages.
#  Every day in one year

import os
import cdsapi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--variable", help="Variable name", type=str, required=True)
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument(
    "--opdir",
    help="Directory for output files",
    default="%s/CPM5/daily/reanalysis" % os.getenv("SCRATCH"),
)
args = parser.parse_args()
args.opdir += "/%04d" % args.year
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir, exist_ok=True)


ctrlB = {
    "format": "netcdf",
    "product_type": "daily_averaged_reanalysis",
    "variable": args.variable,
    "year": ["%04d" % args.year],
    "day": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    "time": "00:00",
}

c = cdsapi.Client()
c.retrieve(
    "reanalysis-era5-single-levels-daily-means",
    ctrlB,
    "%s/%s.nc" % (args.opdir, args.variable),
)

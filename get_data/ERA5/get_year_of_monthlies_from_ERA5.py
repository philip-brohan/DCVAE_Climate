#!/usr/bin/env python

# Retrieve ERA5 monthly averages.
#  Every month in one year

import os
import cdsapi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--variable", help="Variable name", type=str, required=True)
parser.add_argument("--year", help="Year", type=int, required=True)
parser.add_argument(
    "--opdir",
    help="Directory for output files",
    default="%s/ERA5/monthly/reanalysis" % os.getenv("SCRATCH"),
)
args = parser.parse_args()
args.opdir += "/%04d" % args.year
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir, exist_ok=True)


ctrlB = {
    "format": "netcdf",
    "product_type": "monthly_averaged_reanalysis",
    "variable": args.variable,
    "year": ["%04d" % args.year],
    "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    "time": "00:00",
}

c = cdsapi.Client()
c.retrieve(
    "reanalysis-era5-single-levels-monthly-means",
    ctrlB,
    "%s/%s.nc" % (args.opdir, args.variable),
)

# Use system call to run ncks on output file
# To remove 'expver' variable (otherwise iris won't load it)
os.system(
    "ncks -O -C -x -v expver %s/%s.nc %s/%s.nc"
    % (args.opdir, args.variable, args.opdir, args.variable)
)

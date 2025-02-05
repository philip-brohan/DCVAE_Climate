#!/usr/bin/env python

# Get monthly ERA5 data for several years, and store on SCRATCH.

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--startyear", type=int, required=False, default=1940)
parser.add_argument("--endyear", type=int, required=False, default=2024)
args = parser.parse_args()

for year in range(args.startyear, args.endyear + 1):
    for var in [
        "2m_temperature",
        "sea_surface_temperature",
        "mean_sea_level_pressure",
        "total_precipitation",
    ]:
        opfile = "%s/ERA5/monthly/reanalysis/%04d/%s.nc" % (
            os.getenv("SCRATCH"),
            year,
            var,
        )
        if not os.path.isfile(opfile):
            print(
                ("./get_year_of_monthlies_from_ERA5.py --year=%d --variable=%s")
                % (
                    year,
                    var,
                )
            )

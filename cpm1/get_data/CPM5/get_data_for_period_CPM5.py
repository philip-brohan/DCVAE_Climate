#!/usr/bin/env python

# Get daily CPM5 data for several years, and store on SCRATCH.

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--startyear", type=int, required=False, default=1940)
parser.add_argument("--endyear", type=int, required=False, default=2023)
args = parser.parse_args()

for year in range(args.startyear, args.endyear + 1):
    for var in [
        "tas",
        "psl",
        "uas",
        "vas",
    ]:

        opfile = "%s/CPM5/daily/raw/%02d/%s/day/v20210615/%s_rcp85_land-cpm_uk_5km_%02d_day_%04d1201-%04d1130.nc" % (
            os.getenv("SCRATCH"),
            member,
            variable,
            variable,
            member,
            year,
            year+10,
        )
        if not os.path.isfile(opfile):
            print(
                ("./get_year_of_daylies_from_CPM5.py --year=%d --variable=%s")
                % (
                    year,
                    var,
                )
            )

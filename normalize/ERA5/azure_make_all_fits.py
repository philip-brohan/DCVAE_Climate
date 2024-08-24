#!/usr/bin/env python

# Make all the normalization fits

import os

sDir = os.path.dirname(os.path.realpath(__file__))


count = 0
for variable in (
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_precipitation",
    "sea_surface_temperature",
):
    for month in range(1, 13):
        cmd = "%s/azure_fit_for_month.py --month=%02d --variable=%s" % (
            sDir,
            month,
            variable,
        )
        print(cmd)

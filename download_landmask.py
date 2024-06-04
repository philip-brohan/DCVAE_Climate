#!/usr/bin/env python

# Retrieve a soil temperature file from ERA5-land

# This is just an easy way to get a high-resolution land mask for plotting

import os
import cdsapi

opdir ="%s/ERA5/monthly/reanalysis" % os.getenv("SCRATCH")
if not os.path.isdir(opdir):
    os.makedirs(opdir, exist_ok=True)

if not os.path.isfile("%s/land_mask.nc" % opdir): # Only bother if we don't have it

    c = cdsapi.Client()

    # Variable and date are arbitrary
    # Just want something that is only defined in land grid-cells.

    ctrlB = {
        'variable': 'soil_temperature_level_1',
        'year': '2001',
        'month': '03',
        'time': '00:00',
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
    }

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-land-monthly-means",
        ctrlB,
        "%s/%s.nc" % (opdir, 'land_mask'),
    )

#!/bin/bash

# Run all the data downloads

# Retrieves about 8Gb data.
# Will take a few hours - depends on CDS load.

(cd ERA5 && ./get_data_for_period_ERA5.py | parallel -j 1)

(cd land_mask && ./get_land_mask_from_ERA5_land.py)


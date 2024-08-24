#!/bin/bash

# Make all the raw tensors
# Requires downloaded data

(cd ERA5 && ./azure_make_raw_tensors.py --variable=2m_temperature)
(cd ERA5 && ./azure_make_raw_tensors.py --variable=sea_surface_temperature)
(cd ERA5 && ./azure_make_raw_tensors.py --variable=mean_sea_level_pressure)
(cd ERA5 && ./azure_make_raw_tensors.py --variable=total_precipitation)

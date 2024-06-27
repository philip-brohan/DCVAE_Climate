#!/bin/bash

# Make all the normalized tensors
# Requires pre-made normalization parameters.

(cd ERA5 && ./make_all_tensors.py --variable=2m_temperature)
(cd ERA5 && ./make_all_tensors.py --variable=sea_surface_temperature)
(cd ERA5 && ./make_all_tensors.py --variable=mean_sea_level_pressure)
(cd ERA5 && ./make_all_tensors.py --variable=total_precipitation)

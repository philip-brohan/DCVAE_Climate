#!/bin/bash

# Update the metadata for the raw tensors (date::index lists for everything available)
# Make all the tensors first

(cd ERA5 && ./update_tensor_metadata.py --variable=2m_temperature)
(cd ERA5 && ./update_tensor_metadata.py --variable=sea_surface_temperature)
(cd ERA5 && ./update_tensor_metadata.py --variable=mean_sea_level_pressure)
(cd ERA5 && ./update_tensor_metadata.py --variable=total_precipitation)

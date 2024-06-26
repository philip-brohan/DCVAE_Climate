#!/bin/bash

# Make normalization constants for all the datasets
# Requires pre-made raw tensors

(cd ERA5 && ./make_all_fits.py)

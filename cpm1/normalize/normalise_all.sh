#!/usr/bin/bash

# Make normalization constants for all the datasets
# Requires pre-made raw tensors

### normalise_all.sh | ~hadsx/bin/spice_parallel --time=10


(cd CPM5 && ./make_all_fits.py)

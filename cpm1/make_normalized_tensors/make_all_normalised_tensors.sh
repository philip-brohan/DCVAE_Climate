#!/usr/bin/bash

# Make all the normalized tensors
# Requires pre-made normalization parameters.

### make_all_normalised_tensors.sh

# (cd CPM5 && ./make_all_tensors.py --variable=tas)
# (cd CPM5 && ./make_all_tensors.py --variable=psl)
# (cd CPM5 && ./make_all_tensors.py --variable=uas)
(cd CPM5 && ./make_all_tensors.py --variable=vas)

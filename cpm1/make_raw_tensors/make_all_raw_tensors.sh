#!/usr/bin/bash

# Make all the raw tensors
# Requires downloaded data

### make_all_raw_tensors.sh | ~hadsx/bin/spice_parallel --time=5

# (cd CPM5 && ./make_all_tensors.py --member=1 --variable=tas)
(cd CPM5 && ./make_all_tensors.py --member=1 --variable=psl)
sleep 60
(cd CPM5 && ./make_all_tensors.py --member=1 --variable=uas)
sleep 60
(cd CPM5 && ./make_all_tensors.py --member=1 --variable=vas)

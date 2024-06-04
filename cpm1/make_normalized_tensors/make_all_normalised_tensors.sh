#!/usr/bin/bash

# Make all the normalized tensors
# Requires pre-made normalization parameters.

### make_all_normalised_tensors.sh

# echo 'all done 1'
(cd CPM5 && ./make_all_tensors.py --variable=tas)
# echo 'all done 2'
# sleep 60
(cd CPM5 && ./make_all_tensors.py --variable=psl)
# echo 'all done 3'
# sleep 60
(cd CPM5 && ./make_all_tensors.py --variable=uas)
# echo 'all done 4'
# # sleep 60
(cd CPM5 && ./make_all_tensors.py --variable=vas)
# echo 'all done 5'

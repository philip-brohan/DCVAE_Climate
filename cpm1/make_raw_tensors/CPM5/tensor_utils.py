# Utility functions for creating and manipulating raw tensors

import numpy as np
import tensorflow as tf

from get_data.CPM5 import CPM5_daily
from utilities import grids


# Load the data for 1 day (on the standard cube).
def load_raw(member=1, variable="tas", year=1980, day=1):
    raw = CPM5_daily.load(
        member=member,
        variable=variable,
        year=year,
        day=day,
        # grid=grids.OS5sCube,
    )
    raw.data.data[raw.data.mask == True] = np.nan
    return raw


# Convert raw cube to tensor
def raw_to_tensor(raw):
    ict = tf.convert_to_tensor(raw.data, tf.float32)
    return ict


# Convert tensor to cube
def tensor_to_cube(tensor):
    cube = grids.OS5sCube.copy()
    cube.data = tensor.numpy()
    cube.data = np.ma.MaskedArray(cube.data, np.isnan(cube.data))
    return cube

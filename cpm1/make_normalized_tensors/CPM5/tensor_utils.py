# Utility functions for creating and manipulating normalized tensors

import tensorflow as tf
import numpy as np

from get_data.CPM5 import CPM5_daily
from utilities import grids
from normalize.CPM5.normalize import (
    normalize_cube,
    unnormalize_cube,
    load_fitted,
)


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

# Convert raw cube to normalized tensor
def raw_to_tensor(raw, variable, month):
    (shape, location, scale) = load_fitted(month, variable=variable)
    norm = normalize_cube(raw, shape, location, scale)
    norm.data.data[raw.data.mask == True] = 0.0
    ict = tf.convert_to_tensor(norm.data, tf.float32)
    return ict

# Convert normalized tensor to cube
def tensor_to_cube(tensor):
    cube = grids.OS5sCube.copy()
    cube.data = tensor.numpy()
    cube.data = np.ma.MaskedArray(cube.data, cube.data == 0.0)
    return cube


# Convert normalized tensor to raw values
def tensor_to_raw(tensor, variable, month):
    (shape, location, scale) = load_fitted(month, variable=variable)
    cube = tensor_to_cube(tensor)
    raw = unnormalize_cube(cube, shape, location, scale)
    raw.data.data[raw.data.mask == True] = 0.0
    return raw

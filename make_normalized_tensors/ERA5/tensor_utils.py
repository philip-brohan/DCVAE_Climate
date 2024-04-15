# Utility functions for creating and manipulating normalized tensors

import tensorflow as tf
import numpy as np

from get_data.ERA5 import ERA5_monthly
from utilities import grids
from normalize.ERA5.normalize import (
    normalize_cube,
    unnormalize_cube,
    load_fitted,
)


# Load the data for 1 month
def load_raw(year, month, variable="total_precipitation"):
    raw = ERA5_monthly.load(
        variable=variable,
        year=year,
        month=month,
        grid=grids.E5sCube,
    )
    raw.data.data[raw.data.mask == True] = 0.0
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
    cube = grids.E5sCube.copy()
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

# Create raw data dataset for normalization

import os
import sys
import tensorflow as tf
import numpy as np
import zarr
import tensorstore as ts

from make_raw_tensors.ERA5 import tensor_utils


# Get a dataset - all the tensors for a given and variable
def getDataset(
    variable,
    startyear=tensor_utils.FirstYear,
    endyear=tensor_utils.LastYear,
    blur=None,
    cache=False,
):

    # Get the index of the last month in the raw tensors
    fn = "%s/DCVAE-Climate/raw_datasets/ERA5/%s_zarr" % (
        os.getenv("SCRATCH"),
        variable,
    )
    zarr_array = zarr.open(fn, mode="r")
    max_index = zarr_array.shape[2]

    # Get lists of dates and indices
    dates = []
    indices = []
    for i in range(max_index):
        year, month = tensor_utils.index_to_date(i)
        if year >= startyear and year <= endyear:
            dates.append("%04d-%02d" % (year, month))
            indices.append(i)

    # Create TensorFlow Dataset object from the source file dates
    tn_data = tf.data.Dataset.from_tensor_slices(tf.constant(dates))

    # Convert from list of file names to Dataset of source file contents
    tsa = ts.open(
        {
            "driver": "zarr",
            "kvstore": "file://" + fn,
        }
    ).result()

    # Need the indirect function as zarr can't take tensor indices and .map prohibits .numpy()
    def load_tensor_from_index_py(idx):
        return tf.convert_to_tensor(tsa[:, :, idx.numpy()].read().result(), tf.float32)

    def load_tensor_from_index(idx):
        result = tf.py_function(
            load_tensor_from_index_py,
            [idx],
            tf.float32,
        )
        result = tf.reshape(result, [721, 1440, 1])
        return result

    ts_data = tf.data.Dataset.from_tensor_slices(tf.constant(indices, tf.int32))
    ts_data = ts_data.map(
        load_tensor_from_index, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # Add noise to data - needed for some cases where the data is all zero
    if blur is not None:
        ts_data = ts_data.map(
            lambda x: x + tf.random.normal([721, 1440, 1], stddev=blur),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    # Zip the data together with the years (so we can find the date and source of each
    #   data tensor if we need it).
    tz_data = tf.data.Dataset.zip((ts_data, tn_data))

    # Optimisation
    if cache:
        tz_data = tz_data.cache()  # Great, iff you have enough RAM for it

    tz_data = tz_data.prefetch(tf.data.experimental.AUTOTUNE)

    return tz_data

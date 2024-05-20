# Create raw data dataset for normalization

import os
import sys
import tensorflow as tf
import numpy as np


# Load a pre-prepared tensor from a file
def load_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, np.float32)
    imt = tf.reshape(imt, [721, 1440, 1])
    return imt


# Get a list of filenames containing tensors
def getFileNames(variable, startyear=1850, endyear=2050):
    inFiles = sorted(
        os.listdir(
            "%s/DCVAE-Climate/raw_datasets/ERA5/%s" % (os.getenv("SCRATCH"), variable)
        )
    )
    inFiles = [
        fn for fn in inFiles if (int(fn[:4]) >= startyear and int(fn[:4]) <= endyear)
    ]
    return inFiles


# Get a dataset - all the tensors for a given and variable
def getDataset(variable, startyear=1850, endyear=2050, blur=None, cache=False):
    # Get a list of years to include
    inFiles = getFileNames(variable, startyear=startyear, endyear=endyear)

    # Create TensorFlow Dataset object from the source file names
    tn_data = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles))

    # Convert from list of file names to Dataset of source file contents
    fnFiles = [
        "%s/DCVAE-Climate/raw_datasets/ERA5/%s/%s" % (os.getenv("SCRATCH"), variable, x)
        for x in inFiles
    ]
    ts_data = tf.data.Dataset.from_tensor_slices(tf.constant(fnFiles))
    ts_data = ts_data.map(load_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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

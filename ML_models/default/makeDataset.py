# Make tf.data.Datasets from ERA5 monthly averages

# This is a generic script to make a TensorFlow Dataset
# Follow the instructions in autoencoder.py to use it.

import os
import sys
import tensorflow as tf
import numpy as np
import random
import zarr
import tensorstore as ts


# Find out how what months are available from a source
def getDataAvailability(source):
    zfile = "%s/DCVAE-Climate/normalized_datasets/%s_zarr" % (
        os.getenv("SCRATCH"),
        source,
    )
    zarr_array = zarr.open(zfile, mode="r")
    AvailableMonths = zarr_array.attrs["AvailableMonths"]
    return AvailableMonths


# Make a set of months available in all of a set of sources
def getMonths(
    sources,
    purpose,
    firstYr,
    lastYr,
    testSplit,
    maxTrainingMonths,
    maxTestMonths,
):
    avail = {}
    months_in_all = None
    for source in sources:
        avail[source] = getDataAvailability(source)
        if months_in_all is None:
            months_in_all = set(avail[source].keys())
        else:
            months_in_all = months_in_all.intersection(set(avail[source].keys()))

    # Filter by range of years
    filtered = []
    for month in months_in_all:
        year = int(month[:4])
        if (firstYr is None or year >= firstYr) and (lastYr is None or year <= lastYr):
            filtered.append(month)
    months_in_all = filtered

    months_in_all.sort()  # Months in time order (validation plots)

    # Test/Train split
    if purpose is not None:
        test_ns = list(range(0, len(months_in_all), testSplit))
        if purpose == "Train":
            months_in_all = [
                months_in_all[x] for x in range(len(months_in_all)) if x not in test_ns
            ]
        elif purpose == "Test":
            months_in_all = [
                months_in_all[x] for x in range(len(months_in_all)) if x in test_ns
            ]
        else:
            raise Exception("Unsupported purpose " + purpose)

    # Limit maximum data size
    if purpose == "Train" and maxTrainingMonths is not None:
        if len(months_in_all) >= maxTrainingMonths:
            months_in_all = months_in_all[0:maxTrainingMonths]
        else:
            raise ValueError(
                "Only %d months available, can't provide %d"
                % (len(months_in_all), maxTrainingMonths)
            )
    if purpose == "Test" and maxTestMonths is not None:
        if len(months_in_all) >= maxTestMonths:
            months_in_all = months_in_all[0:maxTestMonths]
        else:
            raise ValueError(
                "Only %d months available, can't provide %d"
                % (len(months_in_all), maxTestMonths)
            )

    # Return a list of months
    #  and a list of lists of indices - onje list per source
    indices = {}
    for source in sources:
        indices[source] = []
        for key in months_in_all:
            indices[source].append(avail[source][key])
    return months_in_all, indices


# Get a dataset
def getDataset(specification, purpose):
    # Get a list of months to use - inputs
    inMonths, inIndices = getMonths(
        specification["inputTensors"],
        purpose,
        specification["startYear"],
        specification["endYear"],
        specification["testSplit"],
        specification["maxTrainingMonths"],
        specification["maxTestMonths"],
    )

    # If the outputs are not the same as the inputs, get them too and use only months in both
    if (
        specification["outputTensors"] is not None
    ):  # I.e. input and output are not the same
        outMonths, outIndices = getMonths(
            specification["outputTensors"],
            purpose,
            specification["startYear"],
            specification["endYear"],
            specification["testSplit"],
            specification["maxTrainingMonths"],
            specification["maxTestMonths"],
        )

        outMonths = sorted(
            list(set(inMonths).intersection(set(outMonths)))
        )  # Shared Months
        if len(outMonths) != len(inMonths):
            raise ValueError(
                "Input and output tensors have different months available"
            )  # Deal with this when it becomes a problem

    # Create TensorFlow Dataset object from the date strings
    tnIData = tf.data.Dataset.from_tensor_slices(tf.constant(inMonths))

    # Open all the source tensorstores
    tsa_in = {}
    for source in specification["inputTensors"]:
        zfile = "%s/DCVAE-Climate/normalized_datasets/%s_zarr" % (
            os.getenv("SCRATCH"),
            source,
        )
        tsa_in[source] = ts.open(
            {
                "driver": "zarr",
                "kvstore": "file://" + zfile,
            }
        ).result()

    # Map functions to get tensors from dates and indices
    def load_input_tensors_from_month_py(month):
        mnth = month.numpy().decode("utf-8")
        source = specification["inputTensors"][0]
        tsa = tsa_in[source]
        idx = inIndices[source][inMonths.index(mnth)]
        ima = tf.convert_to_tensor(tsa[:, :, idx].read().result(), tf.float32)
        ima = tf.reshape(ima, [721, 1440, 1])
        for fni in range(1, len(specification["inputTensors"])):
            source = specification["inputTensors"][fni]
            tsa = tsa_in[source]
            idx = inIndices[source][inMonths.index(mnth)]
            imt = tf.convert_to_tensor(tsa[:, :, idx].read().result(), tf.float32)
            imt = tf.reshape(imt, [721, 1440, 1])
            ima = tf.concat([ima, imt], 2)
        return ima

    def load_input_tensor(month):
        result = tf.py_function(
            load_input_tensors_from_month_py,
            [month],
            tf.float32,
        )
        return result

    # Create Dataset from the source file contents
    tsIData = tnIData.map(
        load_input_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if specification["outputTensors"] is not None:
        tsa_out = {}
        for source in specification["outputTensors"]:
            zfile = "%s/DCVAE-Climate/normalized_datasets/%s_zarr" % (
                os.getenv("SCRATCH"),
                source,
            )
            tsa_out[source] = ts.open(
                {
                    "driver": "zarr",
                    "kvstore": "file://" + zfile,
                }
            ).result()

        def load_output_tensors_from_month_py(month):
            mnth = month.numpy().decode("utf-8")
            source = specification["outputTensors"][0]
            tsa = tsa_out[source]
            idx = outIndices[source][outMonths.index(mnth)]
            ima = tf.convert_to_tensor(tsa[:, :, idx].read().result(), tf.float32)
            ima = tf.reshape(ima, [721, 1440, 1])
            for fni in range(1, len(specification["outputTensors"])):
                source = specification["outputTensors"][fni]
                tsa = tsa_out[source]
                idx = outIndices[source][outMonths.index(mnth)]
                imt = tf.convert_to_tensor(tsa[:, :, idx].read().result(), tf.float32)
                imt = tf.reshape(imt, [721, 1440, 1])
                ima = tf.concat([ima, imt], 2)
            return ima

        def load_output_tensor(month):
            result = tf.py_function(
                load_output_tensors_from_month_py,
                [month],
                tf.float32,
            )
            return result

        tsOData = tnIData.map(
            load_output_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    # Zip the data together with the filenames (so we can find the date and source of each
    #   data tensor if we need it).
    if specification["outputTensors"] is not None:
        tz_data = tf.data.Dataset.zip((tnIData, tsIData, tsOData))
    else:
        tz_data = tf.data.Dataset.zip((tnIData, tsIData))

    # Optimisation
    if (purpose == "Train" and specification["trainCache"]) or (
        purpose == "Test" and specification["testCache"]
    ):
        tz_data = tz_data.cache()  # Great, iff you have enough RAM for it

    tz_data = tz_data.prefetch(tf.data.experimental.AUTOTUNE)

    return tz_data

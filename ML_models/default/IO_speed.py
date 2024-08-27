#!/usr/bin/env python

# Same as the autoencoder, except it doesn't do any training,
# just pulls in the data.
# The idea is that this tests the IO speed of the data pipeline.

import os
import sys
import time

# Cut down on the TensorFlow warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

import argparse

# Load the data path, data source, and model specification
from specify import specification
from ML_models.default.makeDataset import getDataset
from ML_models.default.autoencoderModel import DCVAE, getModel


# Get Datasets
def getDatasets():
    # Set up the training data
    trainingData = getDataset(specification, purpose="Train").repeat(5)
    trainingData = trainingData.shuffle(specification["shuffleBufferSize"]).batch(
        specification["batchSize"]
    )
    trainingData = specification["strategy"].experimental_distribute_dataset(
        trainingData
    )
    validationData = getDataset(specification, purpose="Train")
    validationData = validationData.batch(specification["batchSize"])
    validationData = specification["strategy"].experimental_distribute_dataset(
        validationData
    )

    # Set up the test data
    testData = getDataset(specification, purpose="Test")
    testData = testData.shuffle(specification["shuffleBufferSize"]).batch(
        specification["batchSize"]
    )
    testData = specification["strategy"].experimental_distribute_dataset(testData)

    return (trainingData, validationData, testData)


# Instantiate and run the model under the control of the distribution strategy
with specification["strategy"].scope():
    trainingData, validationData, testData = getDatasets()

    autoencoder = getModel(specification)

    # For a few epochs: load the data and report the time taken
    for epoch in range(0, 5):
        start_time = time.time()

        # Load all batches in the training data
        for batch in trainingData:

            per_replica_op = specification["strategy"].run(
                dateStr=batch[0][0].numpy().decode("utf-8")
            )

        end_training_time = time.time()

        # Report time taken
        print("Epoch: {}".format(epoch))
        print(
            "time: {} ".format(
                end_training_time - start_time,
            )
        )

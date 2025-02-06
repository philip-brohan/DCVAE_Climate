#!/usr/bin/env python

# Convolutional Variational Autoencoder.

# This is a generic model that can be used for any set of input and output fields
# To make a specific model, copy this file, specify.py, validate.py, and validate_multi.py
# to a new directory (makeDataset and autoencoderModel are generic - don't copy them).
# Then edit specify.py to choose the input and output fields, and the training parameters.
# Then run this file to train the model, and the validate scripts to test the result.

import os
import sys
import time

# Cut down on the TensorFlow warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epoch", help="Restart from epoch", type=int, required=False, default=1
)
args = parser.parse_args()

# Set the computation strategy
strategy = tf.distribute.get_strategy()
# Set the optimizer
# If restarting (--epoch > 1) the saved optimizer will be used instead
optimizer = tf.keras.optimizers.Adam(5e-4)

# Load the data path, data source, and model specification
from specify import specification
from ML_models.default.makeDataset import getDataset
from ML_models.default.autoencoderModel import getModel


# Get Datasets
def getDatasets():
    # Set up the training data
    trainingData = getDataset(specification, purpose="Train").repeat(1)
    trainingData = trainingData.shuffle(specification["shuffleBufferSize"]).batch(
        specification["batchSize"]
    )
    trainingData = strategy.experimental_distribute_dataset(trainingData)
    validationData = getDataset(specification, purpose="Train")
    validationData = validationData.batch(specification["batchSize"])
    validationData = strategy.experimental_distribute_dataset(validationData)

    # Set up the test data
    testData = getDataset(specification, purpose="Test")
    testData = testData.shuffle(specification["shuffleBufferSize"]).batch(
        specification["batchSize"]
    )
    testData = strategy.experimental_distribute_dataset(testData)

    return (trainingData, validationData, testData)


# Instantiate and run the model under the control of the distribution strategy
with strategy.scope():
    trainingData, validationData, testData = getDatasets()

    autoencoder = getModel(specification, optimizer, epoch=args.epoch)

    # logfile to output the metrics
    log_FN = ("%s/DCVAE-Climate/%s/logs/Training") % (
        os.getenv("SCRATCH"),
        specification["modelName"],
    )
    if not os.path.isdir(os.path.dirname(log_FN)):
        os.makedirs(os.path.dirname(log_FN))
    logfile_writer = tf.summary.create_file_writer(log_FN)
    with logfile_writer.as_default():
        tf.summary.write(
            "OutputNames",
            specification["outputNames"],
            step=0,
        )

    # For each Epoch: train, save state, and report progress
    for epoch in range(args.epoch, specification["nEpochs"] + args.epoch + 1):
        start_time = time.time()

        # Train on all batches in the training data
        for batch in trainingData:
            if specification["trainingMask"] is not None:
                mbatch = tf.where(specification["trainingMask"] != 0, batch[-1], 0.0)
                batch = (batch[:-1], mbatch)
            per_replica_op = strategy.run(
                autoencoder.train_on_batch, args=(batch, None)
            )

        end_training_time = time.time()

        # Validation and output only every printInterval epochs
        if epoch % specification["printInterval"] != 0:
            continue

        # Accumulate average losses over all batches in the validation data
        autoencoder.update_metrics(strategy, validationData, testData)

        # Update the log file with current metrics
        autoencoder.updateLogfile(logfile_writer, epoch)

        end_monitoring_time = time.time()

        # Report progress
        print("Epoch: {}".format(epoch))
        autoencoder.printState()
        print(
            "time: {} (+{})".format(
                int(end_training_time - start_time),
                int(end_monitoring_time - end_training_time),
            )
        )

        # Save model state
        save_dir = "%s/DCVAE-Climate/%s/weights/Epoch_%04d" % (
            os.getenv("SCRATCH"),
            specification["modelName"],
            epoch,
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        autoencoder.save("%s/ckpt.keras" % save_dir, include_optimizer=True)

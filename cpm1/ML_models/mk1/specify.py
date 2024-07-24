# Specification of the model

# As far as possible, everything specific to the model should be in here

# Then the model spec. and dataset input scripts can be generic.
# Follow the instructions in autoencoder.py to use this.

import tensorflow as tf

specification = {}

specification["modelName"] = "mk6" # "rlevAspice" # "mk3"

specification["inputTensors"] = (
    # "ERA5/2m_temperature",
    # "ERA5/mean_sea_level_pressure",
    "CPM5/tasasdas",
    "CPM5/pslsdasd",
    "CPM5/uaswqeee",
    "CPM5/vasrrrrr",

)
# specification["outputTensors"] = (
#     "ERA5/2m_temperature",
#     "ERA5/mean_sea_level_pressure",
#     "ERA5/total_precipitation",
# )  # If None, same as input
specification["outputTensors"] = None

specification["outputNames"] = ['tas','psl','uas','vas']  # For printout

specification["nInputChannels"] = len(specification["inputTensors"])
if specification["outputTensors"] is not None:
    specification["nOutputChannels"] = len(specification["outputTensors"])
else:
    specification["nOutputChannels"] = specification["nInputChannels"]

specification["startYear"] = None  # Start and end years of training period
specification["endYear"] = None  # (if None, use all available)

specification["testSplit"] = 11  # Keep back test case every n months

# Can use less than all the data (for testing)
specification["maxTrainingMonths"] = None
specification["maxTestMonths"] = None

# What to do if there is more than one field/month
specification["maxEnsembleCombinations"] = (
    5  # Every possible combination of ensembles can get large
)
specification["correlatedEnsembles"] = (
    False  # Ensemble member 1 in source 1 matches member 1 in source 2
)

# Fit parameters
specification["nMonthsInEpoch"] = (
    None  # Length of an epoch - if None, use all the data once
)
specification["nEpochs"] = 500  # How many epochs to train for
specification["shuffleBufferSize"] = 1000  # Buffer size for shuffling
specification["batchSize"] = 32  # Arbitrary
specification["beta"]  = 0.05  # Weighting factor for KL divergence of latent space
specification["gamma"] = 0.000  # Weighting factor for KL divergence of output
specification["maxGradient"] = 5  # Numerical instability protection

# Output control
specification["printInterval"] = (
    1  # How often to print metrics and save weights (epochs)
)

# Optimization
specification["strategy"] = tf.distribute.MirroredStrategy()
specification["optimizer"] = tf.keras.optimizers.Adam(5e-5) # tf.keras.optimizers.Adam(5e-4)
specification["trainCache"] = True
specification["testCache"] = True


# Mask to specify a subset of data to train on
specification["trainingMask"] = None  # Train on all data

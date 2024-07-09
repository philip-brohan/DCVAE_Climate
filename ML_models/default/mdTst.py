#!/usr/bin/env python

# Preliminary test of makeDataset.py

import os
import sys
import time

# Cut down on the TensorFlow warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf


# Load the data path, data source, and model specification
from specify import specification
from ML_models.default.makeDataset import getDataset

trainingData = getDataset(specification, purpose=None)

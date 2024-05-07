#!/usr/bin/env python

# Plot validation statistics for all the test cases

from specify import specification

# I don't need all the messages about a missing font
import logging

logging.getLogger("matplotlib.font_manager").disabled = True


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=500)
parser.add_argument(
    "--startyear", help="First year to plot", type=int, required=False, default=None
)
parser.add_argument(
    "--endyear", help="Last year to plot", type=int, required=False, default=None
)
parser.add_argument(
    "--min_lat",
    help="Minimum latitude for target region (-90 to 90)",
    type=float,
    required=False,
    default=-90,
)
parser.add_argument(
    "--max_lat",
    help="Maximum latitude for target region (-90 to 90)",
    type=float,
    required=False,
    default=90,
)
parser.add_argument(
    "--min_lon",
    help="Minimum longitude for target region (-180 to 180)",
    type=float,
    required=False,
    default=-180,
)
parser.add_argument(
    "--max_lon",
    help="Maximum longitude for target region (-180 to 180)",
    type=float,
    required=False,
    default=180,
)
parser.add_argument(
    "--training",
    help="Use training months (not test months)",
    dest="training",
    default=False,
    action="store_true",
)
args = parser.parse_args()

from utilities import grids

from ML_models.default.makeDataset import getDataset
from ML_models.default.autoencoderModel import DCVAE, getModel
from ML_models.default.gmUtils import (
    computeScalarStats,
    plotScalarStats,
)

# Set up the test data
purpose = "Test"
if args.training:
    purpose = "Train"
dataset = getDataset(specification, purpose=purpose)
dataset = dataset.batch(1)

# Load the trained model
autoencoder = getModel(specification, args.epoch)

# Go through the data and get the scalar stat for each test month
all_stats = {}
all_stats["dtp"] = []
all_stats["target"] = {}
all_stats["generated"] = {}
for case in dataset:
    generated = autoencoder.call(case, training=False)
    stats = computeScalarStats(
        specification,
        case,
        generated,
        min_lat=args.min_lat,
        max_lat=args.max_lat,
        min_lon=args.min_lon,
        max_lon=args.max_lon,
    )
    all_stats["dtp"].append(stats["dtp"])
    for key in stats["target"].keys():
        if key in all_stats["target"]:
            all_stats["target"][key].append(stats["target"][key])
            all_stats["generated"][key].append(stats["generated"][key])
        else:
            all_stats["target"][key] = [stats["target"][key]]
            all_stats["generated"][key] = [stats["generated"][key]]

# Make the plot
plotScalarStats(all_stats, specification, fileName="multi.webp")

#!/usr/bin/env python

# Plot time-series of training progress

from specify import specification
import sys
sys.path.append('/home/mo-sbrown/philip1/DCVAE_Climate_sjb1/cpm1')
from ML_models.mk1.gmUtils import loadHistory, plotTrainingMetrics

import argparse

# I don't need all the messages about a missing font
import logging

logging.getLogger("matplotlib.font_manager").disabled = True

parser = argparse.ArgumentParser()
parser.add_argument(
    "--comparator", help="Comparison model name", type=str, required=False, default=None
)
parser.add_argument(
    "--selfc", help="Compare with previous run", type=int, required=False, default=None
)
parser.add_argument(
    "--rscale",
    help="Scale RMS losses in comparator",
    type=float,
    required=False,
    default=1.0,
)
parser.add_argument(
    "--ymax", help="Y range maximum", type=float, required=False, default=None
)
parser.add_argument(
    "--ymin", help="Y range minimum", type=float, required=False, default=None
)
parser.add_argument(
    "--max_epoch", help="Max epoch to plot", type=int, required=False, default=None
)
args = parser.parse_args()

hts = None
chts = None

(hts, ymax, ymin, epoch) = loadHistory(
    specification["modelName"],
)

if args.selfc is not None:
    (chts, cymax, cymin, cepoch) = loadHistory(specification["modelName"], args.selfc)
    epoch = max(epoch, cepoch)
    ymax = max(ymax, cymax)
    ymin = min(ymin, cymin)

if args.comparator is not None:
    (chts, cymax, cymin, cepoch) = loadHistory(args.comparator)
    epoch = max(epoch, cepoch)
    ymax = max(ymax, cymax)
    ymin = min(ymin, cymin)


plotTrainingMetrics(
    specification,
    hts,
    fileName="training.webp",
    chts=chts,
    aymax=args.ymax,
    epoch=epoch,
)

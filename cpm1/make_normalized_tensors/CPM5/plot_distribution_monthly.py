#!/usr/bin/env python

### to use
# module load scitools
# conda activate philip1
# PYTHONPATH=/home/h03/hadsx/extremes/ML/pb1/DCVAE_Climate_sjb1/cpm1
# plot_distribution_monthly.py --year 1980 --day 1 --variable tas

# Plot raw and normalized variable for a selected month
# Map and distribution.

import os
import sys
import numpy as np
import tensorflow as tf

from utilities import plots
from tensor_utils import tensor_to_cube

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import cmocean
import argparse

from scipy.stats import gamma

parser = argparse.ArgumentParser()
parser.add_argument(
    "--year", help="Year to plot", type=int, required=False, default=1980
)
parser.add_argument(
    "--day", help="day to plot", type=int, required=False, default=1
)
parser.add_argument(
    "--variable",
    help="Name of variable to use (tas, psl, ...)",
    type=str,
    default="tas",
)
args = parser.parse_args()


def load_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, np.float32)
    imt = tf.reshape(imt, [244, 180])
    return imt


# Load the fitted values
raw = tensor_to_cube(
    load_tensor(
        "/scratch/hadsx/cpm/5km/daily/1day/%s/raw_tensors/member_01_%04d_%d.tfd"
        % (args.variable, args.year, args.day)
    )
)
normalized = tensor_to_cube(
    load_tensor(
        "/scratch/hadsx/cpm/5km/daily/1day/%s/norm_tensors/member_01_%04d_%d.tfd"
        % (args.variable, args.year, args.day)
    )
)


# Make the plot
fig = Figure(
    figsize=(10 * 3 / 2, 10),
    dpi=100,
    facecolor=(0.5, 0.5, 0.5, 1),
    edgecolor=None,
    linewidth=0.0,
    frameon=False,
    subplotpars=None,
    tight_layout=None,
)
canvas = FigureCanvas(fig)
font = {
    "family": "sans-serif",
    "sans-serif": "Arial",
    "weight": "normal",
    "size": 20,
}
matplotlib.rc("font", **font)
axb = fig.add_axes([0, 0, 1, 1])
axb.set_axis_off()
axb.add_patch(
    Rectangle(
        (0, 0),
        1,
        1,
        facecolor=(1.0, 1.0, 1.0, 1),
        fill=True,
        zorder=1,
    )
)

# # choose actual and normalized data colour maps based on variable
# cmaps = (cmocean.cm.balance, cmocean.cm.balance)
# if args.variable == "total_precipitation":
#     cmaps = (cmocean.cm.rain, cmocean.cm.tarn)
# if args.variable == "mean_sea_level_pressure":
#     cmaps = (cmocean.cm.diff, cmocean.cm.diff)
cmaps = (cmocean.cm.diff, cmocean.cm.diff)

ax_raw = fig.add_axes([0.02, 0.515, 0.607, 0.455])
if args.variable == "total_precipitation":
    vMin = 0
else:
    vMin = np.percentile(raw.data.compressed(), 5)
plots.plotFieldAxes(
    ax_raw,
    raw,
    vMin=vMin,
    vMax=np.percentile(raw.data.compressed(), 95),
    cMap=cmaps[0],
)

ax_hist_raw = fig.add_axes([0.683, 0.535, 0.303, 0.435])
plots.plotHistAxes(ax_hist_raw, raw, bins=25)

ax_normalized = fig.add_axes([0.02, 0.03, 0.607, 0.455])
plots.plotFieldAxes(
    ax_normalized,
    normalized,
    vMin=-0.25,
    vMax=1.25,
    cMap=cmaps[1],
)

ax_hist_normalized = fig.add_axes([0.683, 0.05, 0.303, 0.435])
plots.plotHistAxes(ax_hist_normalized, normalized, vMin=-0.25, vMax=1.25, bins=25)

outfile="pdy2_member_01_%s_%04d-%d.png" % (args.variable, args.year, args.day)
fig.savefig(outfile)

#!/usr/bin/env python

# Check tensor produced as expected

import sys
import numpy as np

from utilities import plots
from tensor_utils import load_raw, raw_to_tensor, tensor_to_cube

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import cmocean
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--member", help="member to plot", type=int, required=False, default=1
)
parser.add_argument(
    "--variable", help="Variable to plot", type=str, required=False, default='tas'
)
parser.add_argument(
    "--day", help="Day to plot", type=int, required=False, default=0
)
parser.add_argument(
    "--year", help="Year to plot", type=int, required=False, default=1980
)
args = parser.parse_args()

raw = load_raw(args.member, args.variable, args.year, args.day)
ict = raw_to_tensor(raw)
ast = tensor_to_cube(ict)

# Make the plot
fig = Figure(
    figsize=(10, 10),
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

ax_o = fig.add_axes([0.05, 0.03, 0.9, 0.45])
plots.plotFieldAxes(
    ax_o,
    ast,
    vMin=np.percentile(np.ma.compressed(ast.data), 0),
    vMax=np.percentile(np.ma.compressed(ast.data), 100),
    cMap=cmocean.cm.balance,
)

ax_r = fig.add_axes([0.05, 0.515, 0.9, 0.45])
plots.plotFieldAxes(
    ax_r,
    raw,
    vMin=np.percentile(np.ma.compressed(raw.data), 0),
    vMax=np.percentile(np.ma.compressed(raw.data), 100),
    cMap=cmocean.cm.balance,
)


fig.savefig("tensor_test.png")

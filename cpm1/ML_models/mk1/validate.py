#!/usr/bin/env python

# Plot a validation figure for the autoencoder.
    #
    # validate.py --epoch 500 --year 1990 --day 786

# exec(open("validate.py").read(), globals())

# For all outputs:
#  1) Target field
#  2) Autoencoder output
#  3) scatter plot

import tensorflow as tf

import sys
sys.path.append('/home/h03/hadsx/extremes/ML/pb1/DCVAE_Climate_sjb1/cpm1')
from ML_models.mk1.makeDataset import getDataset
from ML_models.mk1.autoencoderModel import getModel

from ML_models.mk1.gmUtils import plotValidationField

from specify import specification

import argparse

specification["strategy"] = (
    tf.distribute.get_strategy()
)  # No distribution for simple validation

# I don't need all the messages about a missing font (on Isambard)
import logging

logging.getLogger("matplotlib.font_manager").disabled = True

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=500)
parser.add_argument("--year", help="Test year", type=int, required=False, default=1990)
parser.add_argument(
    "--day", help="Test day", type=int, required=False, default=786
)
parser.add_argument(
    "--training",
    help="Use training data (not test)",
    default=False,
    action="store_true",
)
args = parser.parse_args()

print("Working with model")
print(specification["modelName"])

purpose = "Test"
if args.training:
    purpose = "Train"
# Go through data and get the desired month
dataset = (
    getDataset(specification, purpose=purpose)
    .shuffle(specification["shuffleBufferSize"])
    .batch(1)
)
input = None
year = None
month = None
for batch in dataset:
    dateStr = tf.strings.split(batch[0][0][0], sep="/")[-1].numpy()
    # print(dateStr)
    # year = int(dateStr[:4])
    # month = int(dateStr[5:7])
    year = int(dateStr[10:14]) # int(fN[:4])
    # month = int(fN[5:7])
    idot    = str(dateStr).find('.')
    day     = int(dateStr[15:(idot-2)]) # int(fN[5:7])
    if (day == args.day) and (year == args.year):
        input = batch
        break
    # input = batch

if input is None:
    raise Exception("%04d-%02d not in %s dataset" % (year, day, purpose))

autoencoder = getModel(specification, args.epoch)

# Get autoencoded tensors
output = autoencoder.call(input, training=False)

# Make the plot
title0="%s e%d %04d-%3d" % (specification["modelName"], args.epoch, year, day)
savefile0="val_%s_e%d_%04d-%3d.webp" % (specification["modelName"], args.epoch, year, day)
plotValidationField(specification, input, output, year, day, savefile0, title0)
print("Made plot: ",savefile0)

# ### plot by hand
    # fileName="comparison3.webp"
    # nFields = specification["nOutputChannels"]

    # # Make the plot
    # figScale = 3.0
    # wRatios = (2, 2, 1.25)
    # if specification["trainingMask"] is not None:
    #     wRatios = (2, 2, 1.25, 1.25)
    # fig = Figure(
    #     figsize=(figScale * sum(wRatios), figScale * nFields),
    #     dpi=100,
    #     facecolor=(1, 1, 1, 1),
    #     edgecolor=None,
    #     linewidth=0.0,
    #     frameon=True,
    #     subplotpars=None,
    #     tight_layout=None,
    # )
    # canvas = FigureCanvas(fig)
    # font = {
    #     "family": "DejaVu Sans",
    #     "sans-serif": "Arial",
    #     "weight": "normal",
    #     "size": 12,
    # }
    # matplotlib.rc("font", **font)

    #     # Each variable a row in it's own subfigure
    #     subfigs = fig.subfigures(nFields, 1, wspace=0.01)
    #     if nFields == 1:
    #         subfigs = [subfigs]

    #     for varI in range(nFields):
    #         ax_var = subfigs[varI].subplots(
    #             nrows=1, ncols=len(wRatios), width_ratios=wRatios
    #         )
    #         # Left - map of target
    #         varx = grids.OS5sCube.copy()
    #         varx.data = np.squeeze(input[-1][:, :, :, varI].numpy())
    #         varx.data = np.ma.masked_where(varx.data == 0.0, varx.data, copy=False)
    #         if varI == 0:
    #             # ax_var[0].set_title("%04d-%02d" % (year, day))
    #             ax_var[0].set_title("a plot")
    #         ax_var[0].set_axis_off()
    #         x_img = plots.plotFieldAxes(
    #             ax_var[0],
    #             varx,
    #             vMax=1.25,
    #             vMin=-0.25,
    #             cMap=get_cmap(specification["outputNames"][varI]),
    #         )
    #         # Centre - map of model output
    #         vary = grids.OS5sCube.copy()
    #         vary.data = np.squeeze(output[:, :, :, varI].numpy())
    #         vary.data = np.ma.masked_where(varx.data == 0.0, vary.data, copy=False)
    #         ax_var[1].set_axis_off()
    #         ax_var[1].set_title(specification["outputNames"][varI])
    #         x_img = plots.plotFieldAxes(
    #             ax_var[1],
    #             vary,
    #             vMax=1.25,
    #             vMin=-0.25,
    #             cMap=get_cmap(specification["outputNames"][varI]),
    #         )
    #         # Third - scatter plot of input::output - where used for training
    #         ax_var[2].set_xticks([0, 0.25, 0.5, 0.75, 1])
    #         ax_var[2].set_yticks([0, 0.25, 0.5, 0.75, 1])
    #         if specification["trainingMask"] is not None:
    #             varxm = varx.copy()
    #             varym = vary.copy()
    #             mflat = specification["trainingMask"].numpy().squeeze()
    #             varxm.data = np.ma.masked_where(mflat == 1, varxm.data, copy=True)
    #             varym.data = np.ma.masked_where(mflat == 1, varym.data, copy=True)
    #             plots.plotScatterAxes(
    #                 ax_var[2], varxm, varym, vMin=-0.25, vMax=1.25, bins="log"
    #             )
    #         else:
    #             plots.plotScatterAxes(
    #                 ax_var[2], varx, vary, vMin=-0.25, vMax=1.25, bins="log"
    #             )
    #         # Fourth only if masked - scatter plot of input::output - where masked out of training
    #         if specification["trainingMask"] is not None:
    #             ax_var[3].set_xticks([0, 0.25, 0.5, 0.75, 1])
    #             ax_var[3].set_yticks([0, 0.25, 0.5, 0.75, 1])
    #             mflat = specification["trainingMask"].numpy().squeeze()
    #             varx.data = np.ma.masked_where(mflat == 0, varx.data, copy=True)
    #             vary.data = np.ma.masked_where(mflat == 0, vary.data, copy=True)
    #             plots.plotScatterAxes(
    #                 ax_var[3], varx, vary, vMin=-0.25, vMax=1.25, bins="log"
    #             )

#     fig.savefig(fileName)

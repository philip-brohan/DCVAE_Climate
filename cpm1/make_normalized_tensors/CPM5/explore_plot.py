

#!/usr/bin/env python

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
from matplotlib.pyplot import plt

from tensor_utils import load_raw, raw_to_tensor

import cmocean
import argparse

from scipy.stats import gamma


def load_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, np.float32)
    imt = tf.reshape(imt, [244, 180])
    return imt


t_norm = load_tensor("/scratch/hadsx/cpm/5km/daily/1day/%s/norm_tensors/member_01_%04d_%d.tfd" % ('tas', 1980, 1) )
t_raw  = load_tensor("/scratch/hadsx/cpm/5km/daily/1day/%s/raw_tensors/member_01_%04d_%d.tfd" % ('tas', 1980, 1) )

matplotlib.pyplot.imshow(t_raw)
matplotlib.pyplot.show()

matplotlib.pyplot.hist(t_raw.numpy(), bins=30)
matplotlib.pyplot.xlabel('Values')
matplotlib.pyplot.ylabel('Frequency')
matplotlib.pyplot.title('Basic Histogram')
matplotlib.pyplot.show()


qd = load_raw(member=1, variable='tas', year=1980, day=1)
matplotlib.pyplot.imshow(qd.data)
matplotlib.pyplot.show()

import iris
import iris.util
import iris.coord_systems
n_nc = "/scratch/hadsx/cpm/5km/daily/1day/%s/raw/member_%02d_%04d_%d.nc" % ( 'tas', 1, 1980, 1, )
varC = iris.load_cube(n_nc)
matplotlib.pyplot.imshow(varC.data)
matplotlib.pyplot.show()
cs_CPM5 = iris.coord_systems.RotatedGeogCS(90, 180, 0)
def add_coord_system(cbe):
    cbe.coord("latitude").coord_system = cs_CPM5
    cbe.coord("longitude").coord_system = cs_CPM5

add_coord_system(varC)

s1=10
resolution = 1/s1
xmin = -90/s1
xmax = 90/s1
ymin = -121/s1
ymax = 122/s1
pole_latitude = 90
pole_longitude = 180
npg_longitude = 0
cs = iris.coord_systems.RotatedGeogCS(pole_latitude, pole_longitude, npg_longitude)
lat_values = np.arange(ymin, ymax + resolution, resolution)
latitude = iris.coords.DimCoord(
    lat_values, standard_name="grid_latitude", units="degrees_north", coord_system=cs
)
lon_values = np.arange(xmin, xmax, resolution)
longitude = iris.coords.DimCoord(
    lon_values, standard_name="grid_longitude", units="degrees_east", coord_system=cs
)
dummy_data = np.ma.MaskedArray(np.zeros((len(lat_values), len(lon_values))), False)

OS5sCube = iris.cube.Cube(
    dummy_data, dim_coords_and_dims=[(latitude, 0), (longitude, 1)]
)
grid=OS5sCube

varC = iris.load_cube(n_nc)
varCb = varC.regrid(grid, iris.analysis.Nearest())
matplotlib.pyplot.imshow(varCb.data)
matplotlib.pyplot.show()



#

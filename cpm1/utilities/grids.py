# Define common grids
# Models are grid specific, so it's easier to regrid early on
#  and do everything on the common grid

import numpy as np

import iris
import iris.cube
import iris.util
import iris.analysis
import iris.coord_systems

# Define a standard-cube to work with
# Identical to that used in CPM5, except that the longitude cut is moved
#  to mid pacific (-180) instead of over the UK (0)
resolution = 1
xmin = -90
xmax = 90
ymin = -121
ymax = 122
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
OS5scs = cs

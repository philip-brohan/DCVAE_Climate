# Functions to load CPM5 daily data

# from /spice/project/ukcp/land-cpm/uk/5km/rcp85/01/tas/day/v20210615/

import os
import iris
import iris.util
import iris.coord_systems
import numpy as np

# Don't really understand this, but it gets rid of the error messages.
iris.FUTURE.datum_support = True

# CPM5 data does not have explicit coodinate systems
# Specify one to add on load so the cubes work properly with iris.
cs_CPM5 = iris.coord_systems.RotatedGeogCS(90, 180, 0)


# And a function to add the coord system to a cube (in-place)
def add_coord_system(cbe):
    cbe.coord("latitude").coord_system = cs_CPM5
    cbe.coord("longitude").coord_system = cs_CPM5

def load(
    member=None, variable="tas", year=None, day=None, constraint=None, grid=None
):
    if variable == "land_mask":
        varC = load("sea_surface_temperature", year=2020, day=3, grid=grid)
        varC.data.data[np.where(varC.data.mask == True)] = 0
        varC.data.data[np.where(varC.data.mask == False)] = 1
        return varC
    if year is None or day is None:
        raise Exception("Year and day must be specified") # ll /spice/project/ukcp/land-cpm/uk/5km/rcp85/01/*/day/v20210615/tas_*.nc

    fname = "/scratch/hadsx/cpm/5km/daily/1day/%s/raw/member_%02d_%04d_%d.nc" % (
        variable,
        member,
        year,
        day,
    )
    print("Loading")
    print(fname)
    if not os.path.isfile(fname):
        raise Exception("No data file %s" % fname)
    ftt = iris.Constraint(time=lambda cell: cell.point.day == day)
    # varC = iris.load_cube(fname, ftt)
    varC = iris.load_cube(fname)
    # Get rid of unnecessary height dimensions
    if len(varC.data.shape) == 3:
        varC = varC.extract(iris.Constraint(expver=1))
    add_coord_system(varC)
    varC.long_name = variable
    if grid is not None:
        varC = varC.regrid(grid, iris.analysis.Nearest())
    if constraint is not None:
        varC = varC.extract(constraint)
    return varC

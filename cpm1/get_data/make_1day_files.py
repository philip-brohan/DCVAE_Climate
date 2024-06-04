import os
import iris
import iris.util
import iris.coord_systems
import numpy as np
import datetime
from iris.time import PartialDateTime

# module load scitools
# python -i ./make_1day_files.py

# exec(open('./make_1day_files.py').read())

member=1
stvar='psl'
year=1990

# stin='/spice/project/ukcp/land-cpm/uk/5km/rcp85/01/tas/day/v20210615/tas_rcp85_land-cpm_uk_5km_01_day_19801201-19901130.nc'
stin = "/spice/project/ukcp/land-cpm/uk/5km/rcp85/%02d/%s/day/v20210615/%s_rcp85_land-cpm_uk_5km_%02d_day_%04d1201-%04d1130.nc" % (
    member,
    stvar,
    stvar,
    member,
    year,
    year+10,
)

stdout0='/scratch/hadsx/cpm/5km/daily/1day'
fout0= "%s/%s/raw/member_%02d_%04d" % (
        stdout0,
        stvar,
        member,
        year,
        )


cube0=iris.load_cube(stin)

# tcoord=cube0.coords(standard_name='time')
# print(tcoord)
# for t in tcoord:
#     print(t)

### constrain to summer months
cube1 = cube0.extract(iris.Constraint(time=lambda cell: cell.point.month in [6,7,8]))

tcoord = cube1.coord('time')
# for i, time_point in enumerate(time_coord.points):
for i, time_point in enumerate(tcoord):
    # Extract data for the current date
    time_constr = iris.Constraint(ensemble_member=member, time=lambda t: t.point == tcoord.units.num2date(tcoord.points[i]))

    single_date_cube = cube1.extract(time_constr)

    # Write the single-date cube to a NetCDF file
    output_filename = fout0+"_"+str(i)+".nc"
    # single_date_cube.to_netcdf(output_filename)
    print(output_filename)
    # print(time_point)
    iris.save(single_date_cube, output_filename)

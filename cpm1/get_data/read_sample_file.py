### quick start
# module load scitools	# from 18/3/2019 this will invoke python 3

import numpy as np
import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt

stin='/spice/project/ukcp/land-cpm/uk/5km/rcp85/01/tas/day/v20210615/tas_rcp85_land-cpm_uk_5km_01_day_19801201-19901130.nc'
cube=iris.load_cube(stin)

# # Load your cube (replace 'space_weather.nc' with your actual file)
# filename = iris.sample_data_path(stin)
# cube = iris.load_cube(filename, 'electron density')

# Print the dimension coordinates (e.g., latitude, longitude)
print("Dimension coordinates:")
for coord in cube.coords(dim_coords=True):
    print(f"{coord.name()} {coord.units}")

# Print the auxiliary coordinates (e.g., surface altitude, sigma)
print("\nAuxiliary coordinates:")
for coord in cube.coords(aux_coords=True):
    print(f"{coord.name()} {coord.units}")

# Print the scalar coordinates (e.g., grid latitude, forecast period)
print("\nScalar coordinates:")
for coord in cube.coords(scalar_coords=True):
    print(f"{coord.name()} {coord.points} {coord.units}")


# Get all coordinates in the cube
all_coords = cube.coords()

# Get coordinates with specific attributes (e.g., standard_name, long_name, etc.)
latitude_coords = cube.coords(standard_name='latitude')
longitude_coords = cube.coords(standard_name='longitude')

# Access the coordinate system of latitude
latitude_coord = latitude_coords[0]
coord_system = latitude_coord.coord_system
print(f"Latitude coordinate system: {coord_system}")

print(cube.coord('time').points[0])

>>> max(cube.coord('longitude').points[0])
2.1124915475699564
>>> min(cube.coord('longitude').points[0])
-10.225819972689635

>>> max(cube.coord('latitude').points[0])
49.60721865377003
>>> min(cube.coord('latitude').points[0])
49.31371471134698

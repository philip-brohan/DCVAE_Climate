# Plotting utility functions

import os
import numpy as np

import iris
import iris.util
import iris.analysis
import iris.coord_systems
import iris.exceptions

import matplotlib

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import cmocean

# I don't care about datums.
iris.FUTURE.datum_support = True


# Get the pole location from a cube
#  Assumes an equirectangular projection
def extract_pole(cube):
    try:
        lat = cube.coord("grid_latitude")
        if lat.coord_system is None:
            return (90, 180, 0)
    except Exception:
        return (90, 180, 0)
    if lat.coord_system.grid_mapping_name == "rotated_latitude_longitude":
        return (
            lat.coord_system.grid_north_pole_latitude,
            lat.coord_system.grid_north_pole_longitude,
            lat.coord_system.north_pole_grid_longitude,
        )
    else:
        print(lat.coord_system)
        raise Exception("Unsupported cube for coordinate extraction")


# Make a dummy iris Cube for plotting.
# Makes a cube in equirectangular projection.
# Takes resolution, plot range, and pole location
#  (all in degrees) as arguments, returns an
#  iris cube.
def plot_cube(
    resolution=0.25,
    xmin=-180,
    xmax=180,
    ymin=-90,
    ymax=90,
    pole_latitude=90,
    pole_longitude=180,
    npg_longitude=0,
):
    cs = iris.coord_systems.RotatedGeogCS(pole_latitude, pole_longitude, npg_longitude)
    lat_values = np.arange(ymin, ymax + resolution, resolution)
    latitude = iris.coords.DimCoord(
        lat_values, standard_name="latitude", units="degrees_north", coord_system=cs
    )
    lon_values = np.arange(xmin, xmax + resolution, resolution)
    longitude = iris.coords.DimCoord(
        lon_values, standard_name="longitude", units="degrees_east", coord_system=cs
    )
    dummy_data = np.zeros((len(lat_values), len(lon_values)))
    plot_cube = iris.cube.Cube(
        dummy_data, dim_coords_and_dims=[(latitude, 0), (longitude, 1)]
    )
    return plot_cube


# High res land mask for plots
def get_land_mask(grid_cube=None):
    lm = iris.load_cube(
        "%s/ERA5/monthly/reanalysis/land_mask.nc" % os.getenv("SCRATCH")
    )
    lm = iris.util.squeeze(lm)
    lm.coord("latitude").coord_system = iris.coord_systems.RotatedGeogCS(90, 180, 0)
    lm.coord("longitude").coord_system = iris.coord_systems.RotatedGeogCS(90, 180, 0)
    lm.data = np.where(lm.data.mask, 0, 1)
    if grid_cube is not None:
        lm = lm.regrid(grid_cube, iris.analysis.Linear())
    return lm


# Plot a map in a supplied axes
def plotFieldAxes(
    ax_map,
    field,
    vMax=None,
    vMin=None,
    lMask=None,
    cMap=cmocean.cm.balance,
    plotCube=None,
    f_alpha=1.0,
    show_land=True,
):
    if plotCube is not None:
        field = field.regrid(plotCube, iris.analysis.Linear())
    if vMax is None:
        vMax = np.max(field.data.compressed())
    if vMin is None:
        vMin = np.min(field.data.compressed())
    if lMask is None:
        cs = extract_pole(field)
        lMask = get_land_mask(
            plot_cube(
                resolution=0.1,
                pole_latitude=cs[0],
                pole_longitude=cs[1],
                npg_longitude=cs[2],
            )
        )
    try:
        lons = field.coord("grid_longitude").points
        lats = field.coord("grid_latitude").points
    except iris.exceptions.CoordinateNotFoundError:
        lons = field.coord("longitude").points
        lats = field.coord("latitude").points
    ax_map.set_ylim(min(lats), max(lats))
    ax_map.set_xlim(min(lons), max(lons))
    ax_map.set_axis_off()
    ax_map.set_aspect("equal", adjustable="box", anchor="C")
    ax_map.add_patch(
        Rectangle(
            (min(lons), min(lats)),
            max(lons) - min(lons),
            max(lats) - min(lats),
            facecolor=(0.9, 0.9, 0.9, 1),
            fill=True,
            zorder=1,
        )
    )
    # Plot the field
    T_img = ax_map.pcolorfast(
        lons,
        lats,
        field.data,
        cmap=cMap,
        vmin=vMin,
        vmax=vMax,
        alpha=f_alpha,
        zorder=10,
    )

    # Overlay the land mask
    if show_land:
        mask_img = ax_map.pcolorfast(
            lMask.coord("longitude").points,
            lMask.coord("latitude").points,
            lMask.data,
            cmap=matplotlib.colors.ListedColormap(
                ((0.4, 0.4, 0.4, 0), (0.4, 0.4, 0.4, 0.3))
            ),
            vmin=0,
            vmax=1,
            alpha=1,
            zorder=100,
        )
    return T_img


# Scatter plot in provided axes
def plotScatterAxes(
    ax, var_in, var_out, vMax=None, vMin=None, xlabel="", ylabel="", bins="log"
):
    if vMax is None:
        vMax = max(np.max(var_in.data), np.max(var_out.data))
    if vMin is None:
        vMin = min(np.min(var_in.data), np.min(var_out.data))
    ax.set_xlim(vMin, vMax)
    ax.set_ylim(vMin, vMax)
    ax.hexbin(
        x=var_in.data.compressed(),
        y=var_out.data.compressed(),
        cmap=cmocean.tools.crop_by_percent(cmocean.cm.ice_r, 5, which="min"),
        bins=bins,
        gridsize=50,
        mincnt=1,
    )
    ax.add_line(
        Line2D(
            xdata=(vMin, vMax),
            ydata=(vMin, vMax),
            linestyle="solid",
            linewidth=0.5,
            color=(0.5, 0.5, 0.5, 1),
            zorder=100,
        )
    )
    ax.set(ylabel=ylabel, xlabel=xlabel)
    ax.grid(color="black", alpha=0.2, linestyle="-", linewidth=0.5)


# Histogram in provided axes
def plotHistAxes(ax, var, vMax=None, vMin=None, xlabel="", ylabel="", bins=100):
    if vMax is None:
        vMax = np.max(var.data)
    if vMin is None:
        vMin = np.min(var.data)
    x = var.data.flatten()
    if np.ma.is_masked(x):
        x = x.compressed()
    ax.hist(
        x=x,
        range=(vMin, vMax),
        bins=bins,
        color="blue",
        density=True,
    )
    ax.set(ylabel=ylabel, xlabel=xlabel)
    ax.grid(color="black", alpha=0.2, linestyle="-", linewidth=0.5)

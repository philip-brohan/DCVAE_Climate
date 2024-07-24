# Functions to normalize a data distribution based on SPI
# The aim is to make a normalized distribution that is normally distributed
#  with mean=0.5 and sd=0.2 (so almost all the data is in 0-1)
from scipy.stats import gamma, norm
import numpy as np

import os
import sys
import iris


# Load the pre-calculated fitted values
def load_fitted(month, variable="tas"):
    shape = iris.load_cube(
        "/scratch/hadsx/cpm/5km/daily/1day/%s/normalised/shape_m%02d.nc"
        % (variable, month),
    )
    location = iris.load_cube(
        "/scratch/hadsx/cpm/5km/daily/1day/%s/normalised/location_m%02d.nc"
        % (variable, month),
    )
    scale = iris.load_cube(
        "/scratch/hadsx/cpm/5km/daily/1day/%s/normalised/scale_m%02d.nc"
        % (variable, month),
    )
    return (shape, location, scale)


# Find the normal variate that matches the gamma cdf
def match_normal(raw, gamma_p, norm_mean=0.5, norm_sd=0.2):
    cdf = gamma.cdf(raw, gamma_p[0], gamma_p[1], gamma_p[2])
    cdf[cdf > 0.99999] = 0.99999  # cdf=0 or 1 causes numerical failure
    cdf[cdf < 0.00001] = 0.00001  # Should fix the gamma fit so this never happens
    spi = norm.ppf(cdf, loc=norm_mean, scale=norm_sd)
    return spi


# Find the original value from the normalized one
def match_original(normalized, gamma_p, norm_mean=0.5, norm_sd=0.2):
    cdf = norm.cdf(normalized, loc=norm_mean, scale=norm_sd)
    original = gamma.ppf(cdf, gamma_p[0], gamma_p[1], gamma_p[2])
    return original


# Normalise a cube (same as match_normal but for cubes)
def normalize_cube(raw, shape, location, scale, norm_mean=0.5, norm_sd=0.2):
    cdf = gamma.cdf(raw.data, shape.data, loc=location.data, scale=scale.data)
    cdf[cdf > 0.99999] = 0.99999  # cdf=0 or 1 causes numerical failure
    cdf[cdf < 0.00001] = 0.00001  # Most of these will be missing data
    spi = norm.ppf(cdf, loc=norm_mean, scale=norm_sd)
    result = raw.copy()
    result.data = np.ma.MaskedArray(spi, np.logical_and(raw.data.mask, shape.data.mask))
    result.data.data[result.data.mask] = 0.0
    return result


# Convert a cube from normalized value to raw
#  (same as match_original but for cubes)
def unnormalize_cube(normalized, shape, location, scale, norm_mean=0.5, norm_sd=0.2):
    cdf = norm.cdf(normalized.data, loc=norm_mean, scale=norm_sd)
    raw = gamma.ppf(cdf, shape.data, location.data, scale.data)
    result = normalized.copy()
    result.data.data = raw
    return result

#!/usr/bin/env python

# Retrieve a soil temperature file from ERA5-land

# This is just an easy way to get a high-resolution land mask for plotting

import os
import tempfile
import cdsapi
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import DataImport
from azure.ai.ml.data_transfer import Database
from azure.ai.ml import MLClient

opdir ="%s/ERA5/monthly/reanalysis" % os.getenv("SCRATCH")
#if not os.path.isdir(opdir):
#    os.makedirs(opdir, exist_ok=True)

#if not os.path.isfile("%s/land_mask.nc" % opdir): # Only bother if we don't have it

c = cdsapi.Client()

# Variable and date are arbitrary
# Just want something that is only defined in land grid-cells.

ctrlB = {
    'variable': 'soil_temperature_level_1',
    'year': '2001',
    'month': '03',
    'time': '00:00',
    'format': 'netcdf',
    'product_type': 'monthly_averaged_reanalysis',
}

with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as temp_file:
    temp_filename = temp_file.name


r = c.retrieve(
    "reanalysis-era5-land-monthly-means",
    ctrlB,
    temp_filename,
)


# Move to datastore
ml_client = MLClient.from_config(credential = DefaultAzureCredential())
try:
    data_import = DataImport(
        name="land_mask",
        source={"type": "file_system","path": temp_filename},
        path="azureml://datastores/%s/%s.nc" % ('workspaceblobstore/paths/Philip_SCRATCH', 'land_mask')
        )
    ml_client.data.import_data(data_import=data_import)
except Exception as e:
    print(e)

os.remove(temp_filename)



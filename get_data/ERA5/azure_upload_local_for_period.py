#!/usr/bin/env python

# Upload ERA% data to Azure Data Lake

import os
import argparse
import warnings

from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    DeviceCodeCredential,
)
from azure.ai.ml import MLClient

from azure.storage.filedatalake import (
    DataLakeServiceClient,
    DataLakeDirectoryClient,
    FileSystemClient,
)

parser = argparse.ArgumentParser()
parser.add_argument("--startyear", type=int, required=False, default=1940)
parser.add_argument("--endyear", type=int, required=False, default=1940)
args = parser.parse_args()

# Connect using Default Credential - dependent on already being logged in via Azure CLI in the current environment
try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    token = credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    print("using fallback")
    # credential = InteractiveBrowserCredential()
    ## does not work when working from Azure Databricks as we cannot launch a browser
    credential = DeviceCodeCredential(tenant_id=os.environ["AZURE_TENANT_ID"])
    ## this may work if conditional access controlling login device types is not set by your tenant admin

# set up the mlclient
ml_client = MLClient(
    credential=credential,
    subscription_id=os.environ.get("AZML_SUBSCRIPTION_ID"),
    resource_group_name=os.environ.get("AZML_RESOURCE_GROUP"),
    workspace_name=os.environ.get("AZML_WORKSPACE_NAME"),
)

# Where from and where to
local_root = "/scratch/hadpb/ERA5/monthly/reanalysis"
storage_account_name = "dcvaelake"
file_system_name = "copper"
target_root = "SCRATCH/ERA5/monthly/reanalysis"

# Connect to the storage account
service_client = DataLakeServiceClient(
    "https://%s.dfs.core.windows.net" % storage_account_name,
    credential=credential,
)
# Check it exists
try:
    sap = service_client.get_service_properties()
except Exception as e:
    print("Storage account %s not found" % storage_account_name)

# Get the file system
file_system_client = service_client.get_file_system_client(file_system_name)
# Check it exists
if not file_system_client.exists():
    raise Exception("File system %s not found" % file_system_name)

# Get the target root directory - make it if it doesn't exist
directory_client = file_system_client.get_directory_client(target_root)
if not directory_client.exists():
    directory_client = file_system_client.create_directory(target_root)

# Loop over the years
for year in range(args.startyear, args.endyear + 1):
    directory_client = file_system_client.get_directory_client(
        "%s/%04d" % (target_root, year)
    )
    if not directory_client.exists():
        directory_client = file_system_client.create_directory(
            "%s/%04d" % (target_root, year)
        )

    # Loop over the variables
    for var in [
        "2m_temperature",
        "sea_surface_temperature",
        "mean_sea_level_pressure",
        "total_precipitation",
    ]:
        # Get the file
        local_file = "%s/%04d/%s.nc" % (local_root, year, var)
        if not os.path.exists(local_file):
            warnings.warn("File %s does not exist locally" % local_file)
            continue
        local_directory_name, file_name = os.path.split(local_file)
        file_client = directory_client.get_file_client(file_name)
        # If it already exists, we're done
        if file_client.exists():
            continue
        # Upload the file
        with open(local_file, "rb") as data:
            file_client.upload_data(data, overwrite=True)

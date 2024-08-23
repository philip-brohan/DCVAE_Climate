#!/usr/bin/env python

# Upload a file to a given location in an Azure hierachical data lake

import os
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

# File to upload
local_file = "/scratch/hadpb/ERA5/monthly/reanalysis/land_mask.nc"
local_directory_name, file_name = os.path.split(local_file)

# Target storage account
storage_account_name = "dcvaelake"

# Target file system (container on that storage account)
file_system_name = "copper"

# Target directory
directory_name = "ERA5/monthly/reanalysis"

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

# Get the directory - make it if it doesn't exist
directory_client = file_system_client.get_directory_client(directory_name)
if not directory_client.exists():
    directory_client = file_system_client.create_directory(directory_name)

# Get the file
file_client = directory_client.get_file_client(file_name)
# If it already exists, we're done
if file_client.exists():
    raise Exception("File %s/%s already exists" % (directory_name, file_name))

# Upload the file
with open(local_file, "rb") as data:
    file_client.upload_data(data, overwrite=True)

# Download should work too
#    with open(file=os.path.join(local_path, file_name), mode="wb") as local_file:
#        download = file_client.download_file()
#        local_file.write(download.readall())
#        local_file.close()

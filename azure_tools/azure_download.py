#!/usr/bin/env python

# Download files from an Azure Data Lake

import os
import sys
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
parser.add_argument("--local", type=str, help="Local directory", required=True)
parser.add_argument("--remote", type=str, help="Remote directory", required=True)
parser.add_argument(
    "--storage_account",
    type=str,
    help="Storage account name",
    required=False,
    default="dcvaelake",
)
parser.add_argument(
    "--file_system", type=str, help="File system name", required=False, default="copper"
)
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
args = parser.parse_args()

# Connect using Default Credential - dependent on already being logged in via Azure CLI in the current environment
try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    token = credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    print("Login with az login before running this script")

# set up the mlclient
ml_client = MLClient(
    credential=credential,
    subscription_id=os.environ.get("AZML_SUBSCRIPTION_ID"),
    resource_group_name=os.environ.get("AZML_RESOURCE_GROUP"),
    workspace_name=os.environ.get("AZML_WORKSPACE_NAME"),
)

# Connect to the storage account
service_client = DataLakeServiceClient(
    "https://%s.dfs.core.windows.net" % args.storage_account,
    credential=credential,
)
# Check it exists
try:
    sap = service_client.get_service_properties()
except Exception as e:
    print("Storage account %s not found" % args.storage_account)


# Get the file system
file_system_client = service_client.get_file_system_client(args.file_system)
# Check it exists
if not file_system_client.exists():
    raise Exception("File system %s not found" % args.file_system)

# Get paths of all files and directories under args.remote
file_path = file_system_client.get_paths(args.remote, recursive=True)

# Do the downloads
for f in file_path:
    # Make local name from remote name
    if f.name.startswith(args.remote):
        local_name = f.name[len(args.remote) :]
    else:
        raise Exception("File %s not in remote root" % f.name)
    if local_name.startswith("/"):
        local_name = local_name[1:]
    local_name = args.local + "/" + local_name

    if f.is_directory:
        # Make the directory
        if not os.path.exists(local_name):
            os.makedirs(local_name)
    else:  # f is a file
        # If it already exists locally, we're done unless we want to overwrite it
        if os.path.exists(local_name) and not args.overwrite:
            pass
        else:
            # Download the file
            with open(local_name, "wb") as data:
                file_client = file_system_client.get_file_client(f.name)
                file_client.download_file().readinto(data)

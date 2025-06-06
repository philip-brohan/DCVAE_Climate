#!/usr/bin/env python

# Upload files to an Azure Data Lake

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
parser.add_argument("--show_uploaded", action="store_true", help="Show uploaded files")
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


# Upload a file - called recursively if a directory
def upload_file(file_name, file_system_client):
    # Make remote name from local name
    if file_name.startswith(args.local):
        remote_name = file_name[len(args.local) :]
    else:
        raise Exception("File %s not in local root" % file_name)
    if remote_name.startswith("/"):
        remote_name = remote_name[1:]
    remote_name = args.remote + "/" + remote_name

    if os.path.isdir(file_name):
        # Make the directory
        directory_client = file_system_client.get_directory_client(remote_name)
        if not directory_client.exists():
            directory_client = file_system_client.create_directory(remote_name)
        # Loop over the files in the directory
        for f in sorted(os.listdir(file_name)):
            upload_file(os.path.join(file_name, f), file_system_client)
    else:
        # Upload the file
        file_client = file_system_client.get_file_client(remote_name)
        # If it already exists, we're done unless we want to overwrite it
        # Also re-upload zero size files - upload failures leave zero size files
        if file_client.exists() and not args.overwrite:
            properties = file_client.get_file_properties()
            if properties["size"] != 0:
                if args.show_uploaded:
                    print("File %s already uploaded" % file_name)
                return
        # Upload the file
        with open(file_name, "rb") as data:
            try:
                file_client.upload_data(data, overwrite=True)
                print("Uploaded %s" % file_name)
            except Exception as e:
                print("Error uploading %s" % file_name)


# Get the file system
file_system_client = service_client.get_file_system_client(args.file_system)
# Check it exists
if not file_system_client.exists():
    raise Exception("File system %s not found" % args.file_system)

# Do the uploads
upload_file(args.local, file_system_client)

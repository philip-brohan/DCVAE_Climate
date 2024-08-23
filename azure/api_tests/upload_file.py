#!/usr/bin/env python

# This script makes a compute environment from a conda yml file

import os
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    DeviceCodeCredential,
)
from azure.ai.ml import MLClient

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

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


try:
    registered_data_asset = ml_client.data.get(name="Land_Mask", version="1")
    print("Found data asset. Will not create again")
except Exception as ex:
    my_data = Data(
        path="/scratch/hadpb/ERA5/monthly/reanalysis/land_mask.nc",
        type=AssetTypes.URI_FILE,
        description="ERA5 land mask nc file",
        name="Land_Mask",
        version="1",
    )
    ml_client.data.create_or_update(my_data)
    registered_data_asset = ml_client.data.get(name="Land_Mask", version="1")
    print("Created data asset")

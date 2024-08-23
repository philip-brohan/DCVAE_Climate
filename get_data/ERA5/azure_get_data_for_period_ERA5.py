#!/usr/bin/env python

# This script runs the job specified in get_land_mask_azjob.yml on azure

import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--startyear", type=int, required=False, default=1940)
parser.add_argument("--endyear", type=int, required=False, default=2023)
args = parser.parse_args()

from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    DeviceCodeCredential,
)
from azure.ai.ml import MLClient
from azure.ai.ml import command, Output
from azure.ai.ml.constants import AssetTypes, InputOutputModes

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

# Get ~/.cdsapirc - so we can put it on the Azure machine
with open("%s/.cdsapirc" % os.getenv("HOME"), "r") as f:
    cdsapirc = f.read()

# define the job
command_job = command(
    name="get_data_era5_%s" % datetime.now().strftime("%Y%m%d%H%M%S"),
    compute="cpu-cluster",
    environment="DCVAE-Azure@latest",
    code="/net/home/h03/hadpb/Projects/DCVAE_Climate",
    outputs={
        "SCRATCH": Output(
            type=AssetTypes.URI_FOLDER,
            path="azureml://subscriptions/ef7c87c5-ea27-4e06-9fad-ac06dc6b3fd1/resourcegroups/rg-AI4-Climate/workspaces/ai4climate-scratch/datastores/dcvaelake_bronze/paths/SCRATCH/",
            mode=InputOutputModes.RW_MOUNT,
        )
    },
    environment_variables={"SCRATCH": "${{outputs.SCRATCH}}", "CDSAPIRC": cdsapirc},
    command='echo -e "$CDSAPIRC" > /root/.cdsapirc; python get_data/ERA5/get_data_for_period_ERA5.py --startyear %d --endyear %d | parallel -j 1'
    % (args.startyear, args.endyear),
)

# Submit the job
returned_job = ml_client.jobs.create_or_update(command_job)
# get a URL for the status of the job
print(returned_job.studio_url)

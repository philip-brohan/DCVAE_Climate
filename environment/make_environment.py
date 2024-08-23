#!/usr/bin/env python

# This script makes a compute environment from a conda yml file

import os
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    DeviceCodeCredential,
)
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment


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

# define the environment from a conda yml file
bindir = os.path.abspath(os.path.dirname(__file__))
env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file="%s/environment/DCVAE-Climate.yml" % bindir,
    name="DCVAE-Azure",
    description="DCVAE environment: Docker image plus Conda environment.",
)
# ml_client.environments.create_or_update(env_docker_conda)

print("\nAvailable environments (examples):")
for e in ml_client.environments.list():
    if (
        e.creation_context.created_by_type == "User"
        and e.creation_context.created_by != "Microsoft"
    ):
        print(e.name, e.latest_version)

# Note. You can't delete an environment - archive them instead.

# e.g. ml_client.environments.archive('DCVAE-Azure_1')

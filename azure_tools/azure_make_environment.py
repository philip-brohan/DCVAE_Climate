#!/usr/bin/env python

# This script makes an azure compute environment from a conda yml file

import os
from azure.identity import (
    DefaultAzureCredential,
)
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment


# Connect using Default Credential
try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    token = credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    raise Exception("Login with `az login' before running this script")

# Authenticate
ml_client = MLClient(
    credential=credential,
    subscription_id=os.environ.get("AZML_SUBSCRIPTION_ID"),
    resource_group_name=os.environ.get("AZML_RESOURCE_GROUP"),
    workspace_name=os.environ.get("AZML_WORKSPACE_NAME"),
)

# Define the environment from a conda yml file
bindir = os.path.abspath(os.path.dirname(__file__))
env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file="%s/../environment/DCVAE-Climate.yml" % bindir,
    name="DCVAE-Azure",
    description="DCVAE environment: Docker image plus Conda environment.",
)

# We need to do the image build on one of our compute instances, not serverless.
# ws = ml_client.workspaces.get(
#     name=os.environ.get("AZML_WORKSPACE_NAME"),
# )
# ws.image_build_compute = "Basic"  # Build the environment on the Basic compute instance
# ml_client.workspaces.begin_update(ws)

# Create or update the environment in the workspace
ml_client.environments.create_or_update(env_docker_conda)

# print("\nAvailable environments (examples):")
# for e in ml_client.environments.list():
#    if (
#        e.creation_context.created_by_type == "User"
#        and e.creation_context.created_by != "Microsoft"
#    ):
#        print(e.name, e.latest_version)

# Note. You can't delete an environment - archive them instead.

# e.g. ml_client.environments.archive('MLP-Azure')

#!/usr/bin/env python

# Run the given script on azure

import os
import sys
import argparse
from datetime import datetime

from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    DeviceCodeCredential,
)
from azure.ai.ml import MLClient
from azure.ai.ml import command, Output, Input
from azure.ai.ml.constants import AssetTypes, InputOutputModes

# Command to be run is everything in the input after the '--'
#  or everything except the first command if there is no '--'
try:
    command_index = sys.argv.index("--") + 1
except ValueError:
    command_index = 1
cmd = " ".join(sys.argv[command_index:])

name = sys.argv[command_index]
name = "".join(
    c if c.isalnum() else "-" for c in name
)  # remove any non-alphanumeric characters

# Need to add the path to the script from the PYTHONPATH
bindir = os.getcwd()
main_dir = "DCVAE_Climate"
idx = bindir.find(main_dir)
cmd = "%s/%s" % (
    bindir[(idx + len(main_dir) + 1) :],
    cmd,
)

# Script directory contains a lot of crap that doesn't need to be sent to Azure
try:
    ignore_file = "%s/%s" % (
        bindir[(idx + len(main_dir) + 1) :],
        ".amlignore",
    )
except FileNotFoundError:
    ignore_file = None


# only look at the bit before to '--' for arguments to this script
if command_index > 1:
    sys.argv = sys.argv[: (command_index - 1)]
else:
    sys.argv = sys.argv[:1]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--compute", help="Compute to use", type=str, required=False, default="Basic"
)
parser.add_argument("--name", help="Job name", type=str, required=False, default=None)
parser.add_argument(
    "--experiment",
    help="Experiment name",
    type=str,
    required=False,
    default="Unspecified",
)
parser.add_argument(
    "--dryrun", help="Print YML instead of submitting", action="store_true"
)
parser.add_argument(
    "--parallel", help="Run outputs in parallel", type=int, default=None
)
args = parser.parse_args()

if args.parallel is not None:
    cmd = "%s | parallel -j %d" % (cmd, args.parallel)
cmd = "%s > logs/output.txt" % cmd

# Connect using Default Credential - dependent on already being logged in via Azure CLI in the current environment
try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    token = credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    print("Can't authenticate - maybe not logged in via Azure CLI")

# set up the mlclient
ml_client = MLClient(
    credential=credential,
    subscription_id=os.environ.get("AZML_SUBSCRIPTION_ID"),
    resource_group_name=os.environ.get("AZML_RESOURCE_GROUP"),
    workspace_name=os.environ.get("AZML_WORKSPACE_NAME"),
)

# define the job
if args.name is None:
    args.name = name
# args.name = "%s_%s" % (args.name, datetime.now().strftime("%Y%m%d%H%M%S"))
command_job = command(
    name=args.name,
    experiment_name=args.experiment,
    compute=args.compute,
    environment="DCVAE-Azure@latest",
    code="/home/users/philip.brohan/Projects/DCVAE_Climate",
    ignore_file=ignore_file,
    outputs={
        "PDIR": Output(
            type=AssetTypes.URI_FOLDER,
            path=(
                "azureml://subscriptions/%s/"
                + "resourcegroups/%s/workspaces/%s/"
                + "datastores/large_datastore/paths/projects/DCVAE-Climate"
            )
            % (
                os.getenv("AZML_SUBSCRIPTION_ID"),
                os.getenv("AZML_RESOURCE_GROUP"),
                os.getenv("AZML_WORKSPACE_NAME"),
            ),
            mode=InputOutputModes.RW_MOUNT,
        ),
    },
    environment_variables={
        "PDIR": "${{outputs.PDIR}}",
        "AZUREML_FLUSH_INTERVAL": "10",  # Flush output every 10 seconds
        "PYTHONUNBUFFERED": "1",  # Keep Python from buffering output
    },
    command="export PYTHONPATH=$(pwd):$PYTHONPATH ; %s" % cmd,
)

if args.dryrun:
    print(command_job._to_yaml())
    sys.exit(0)

# Submit the job
returned_job = ml_client.jobs.create_or_update(command_job, stream=True)
# get a URL for the status of the job
print(returned_job.studio_url)

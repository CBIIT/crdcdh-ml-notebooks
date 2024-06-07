import json
import os
from pathlib import Path


def get_current_folder(global_variables):
    # if calling from a file
    if "__file__" in global_variables:
        current_file = Path(global_variables["__file__"])
        current_folder = current_file.parent.resolve()
    # if calling from a notebook
    else:
        current_folder = Path(os.getcwd())
    return current_folder.parent

current_folder = get_current_folder(globals())
print("src dir:" + str(current_folder))
ROOT_DIR = current_folder.parent.resolve()
print("root dir:" + str(ROOT_DIR)) 
TIER = "dev"  # tier value in [loc, dev, qa, stage, prod]
CONTAINER_IMAGE_NAME = "blazingtext"
CONTAINER_IMAGE_VERSION = "latest"

CRDCDH_S3_BUCKET = "crdcdh-ml-" + TIER
TRAIN_DATA_PREFIX = "data/train/" + CONTAINER_IMAGE_NAME + "-"
TEST_DATA_PREFIX = "data/test/" + CONTAINER_IMAGE_NAME + "-"
RAW_DATA_PREFIX = "data/raw/json/"
TRAIN_OUTPUTS_PREFIX = "train_output/"

TRAINING_INSTANCE_TYPE = "ml.c4.xlarge"
HOSTING_INSTANCE_TYPE = "ml.m4.xlarge"

SOLUTION_PREFIX = "crdcdh-ml"
TAG_KEY = SOLUTION_PREFIX + "-" + TIER
ENDPOINT_NAME = TAG_KEY + "-endpoint"

# execute-role for sagemaker in FNL for local development only
SAGEMAKER_EXECUTE_ROLE = ""

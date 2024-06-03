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
DATASETS_S3_PREFIX = "datasets"
OUTPUTS_S3_PREFIX = "outputs"

# model source s3 configurations
SOURCE_S3_PREFIX = ""
SOURCE_S3_BUCKET = ""
SOURCE_S3_PATH = f"s3://{SOURCE_S3_BUCKET}/{SOURCE_S3_PREFIX}"

SOLUTION_PREFIX = "crdcdh-ml"

# training and hosting instance type
TRAINING_INSTANCE_TYPE = "ml.c4.xlarge"
HOSTING_INSTANCE_TYPE = "ml.t3.medium"

TAG_KEY = "crdcdh-ml-dev"
# execute-role for sagemaker
SAGEMAKER_EXECUTE_ROLE = ""


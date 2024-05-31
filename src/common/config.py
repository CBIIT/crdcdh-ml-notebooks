import json
import os
from pathlib import Path


def get_current_folder(global_variables):
    # if calling from a file
    if "__file__" in global_variables:
        current_file = Path(global_variables["__file__"])
        current_folder = current_file.parent.parent.resolve()
    # if calling from a notebook
    else:
        current_folder = Path(os.getcwd()).parent
    return current_folder

current_folder = get_current_folder(globals())
print("src dir:" + str(current_folder))
ROOT_DIR = current_folder.parent.resolve()
print("root dir:" + str(ROOT_DIR)) 
DATASETS_S3_PREFIX = "datasets"
OUTPUTS_S3_PREFIX = "outputs"

SOURCE_S3_PREFIX = "ml-data"
SOURCE_S3_BUCKET = "crdcdh-test-submission"
SOURCE_S3_PATH = f"s3://{SOURCE_S3_BUCKET}/{SOURCE_S3_PREFIX}"

SOLUTION_PREFIX = "myTest"

TRAINING_INSTANCE_TYPE = "ml.t3.medium"
HOSTING_INSTANCE_TYPE = "ml.t3.medium"

TAG_KEY = "crdcdh-ml-dev"

SAGEMAKER_EXECUTE_ROLE = "arn:aws:iam::420434175168:role/service-role/AmazonSageMaker-ExecutionRole-20240524T161395"

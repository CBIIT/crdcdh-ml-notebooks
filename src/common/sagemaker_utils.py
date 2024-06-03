import os
from sagemaker.session import Session
from sagemaker.config import load_sagemaker_config, validate_sagemaker_config
from sagemaker.local import LocalSession
import common.sagemaker_config_sample as sagemaker_config_sample

def get_sagemaker_session(is_Local = False):
    if not is_Local: 
        # Create a configuration dictionary manually
        custom_sagemaker_config = load_sagemaker_config(
            additional_config_paths=[
                os.path.join(sagemaker_config_sample.ROOT_DIR, "configs/sagemaker/config.yaml")
            ]
        )
        # Then validate that the dictionary adheres to the configuration schema
        validate_sagemaker_config(custom_sagemaker_config)

        # Then initialize the Session object with the configuration dictionary
        return Session(
            sagemaker_config = custom_sagemaker_config
        )
    
    else:
        sagemaker_session = LocalSession()
        sagemaker_session.config = {'local': {'local_code': True, 'region_name': "us-east-1" }}
        return sagemaker_session

import os
import glob
import sagemaker
import boto3
import json

import sagemaker.deserializers
import sagemaker.serializers
from bento.common.s3 import S3Bucket
from common.sagemaker_utils import get_sagemaker_session, delete_endpoint, get_sagemaker_execute_role, create_local_output_dirs
import common.sagemaker_config as config
from common.utils import untar_file, pretty_print_json, get_data_time
from common.visualize_word_vecs import plot_word_vecs_tsne
from common.transform_json import transform_json_to_training_data

class SemanticAnalysis:
    def __init__(self):
        """
        constructor of SemanticAnalysis
        """
        create_local_output_dirs(["../output", "../output/model", "../data", "../data/raw", "../data/raw/json","../temp"]) # create local directories if not exist.
        self.endpoint_name = None
        self.S3Bucket = S3Bucket(config.CRDCDH_S3_BUCKET)
        self.timestamp = get_data_time()
        self.data_time = get_data_time("%Y-%m-%d-%H-%M-%S")
        self.session = get_sagemaker_session()
        self.role = get_sagemaker_execute_role(self.session)
        # print(self.role)  # This is the role that SageMaker would use to leverage AWS resources (S3, CloudWatch) on your behalf
        self.region_name = boto3.Session().region_name
        self.output_bucket = config.CRDCDH_S3_BUCKET
        self.raw_data_prefix = config.RAW_DATA_PREFIX
        self.train_data_prefix = config.TRAIN_DATA_PREFIX + self.timestamp
        self.test_data_prefix = config.TEST_DATA_PREFIX + self.data_time
        self.output_prefix = config.TRAIN_OUTPUTS_PREFIX
        self.s3_output_location = f"s3://{self.output_bucket}/{self.output_prefix}"
        self.image_name = None
        self.container = None
        self.s3_train_data = None
        self.bt_model = None
        self.data_channels = None
        self.bt_endpoint = None

    def get_s3_bucket(self):
        """
        get s3 bucket
        """
        return self.S3Bucket
        
    def set_container(self, image_name=None, version=None):
        """
        download the container image from ECR
        """
        if image_name is None:
            image_name = config.CONTAINER_IMAGE_NAME
        self.image_name = image_name
        if version is None:
            version = config.CONTAINER_IMAGE_VERSION
        try:   
            self.container = sagemaker.amazon.amazon_estimator.get_image_uri(self.region_name, image_name, version)
            print(f"Using SageMaker BlazingText container: {self.container} ({self.region_name})")
        except Exception as e:
            print("Failed to download container, please contact admin.")
            self.close()

    def transformData(self, s3_raw_data_prefix):
        """
        transform the raw data in json format to training data in text8
        """
        local_raw_data_folder = "../data/raw/json/"
        local_train_data_folder = "../data/train/"
        s3_training_data_file_key = os.path.join(config.TRAIN_DATA_PREFIX, f"{self.image_name}-{self.timestamp}/train_data_text8")
        try:
            if not s3_raw_data_prefix:
                #check if json file in local folder
                if not os.path.exists(local_raw_data_folder) or not any(fname.endswith('.json') for fname in os.listdir(local_raw_data_folder)):
                    print("No raw data found on s3 bucket or local folder")
                    self.close()
                else:
                    # if json file in local folder, upload to designated s3 bucket
                    s3_raw_data_prefix = os.path.join(config.RAW_DATA_PREFIX, f"{self.image_name}-{self.data_time}/")
                    print(f"Upload raw data to s3 bucket, {s3_raw_data_prefix}.")

                    json_files = glob.glob('{}/*.json'.format(local_raw_data_folder))
                    for file in json_files:
                        file_name = os.path.basename(file)
                        self.S3Bucket.upload_file(os.path.join(s3_raw_data_prefix,file_name), file) 
                    print(f"Raw data uploaded to s3 bucket, {s3_raw_data_prefix}.")

            local_raw_data_folder_path = os.path.join(local_raw_data_folder, f"{self.image_name}-{self.data_time}/")
            local_train_data_folder_path = os.path.join(local_train_data_folder, f"{self.image_name}-{self.data_time}/")
            transform_json_to_training_data(s3_raw_data_prefix, local_raw_data_folder_path, s3_training_data_file_key, local_train_data_folder_path)
            return s3_training_data_file_key
        except Exception as e:
            print(f"Failed to transform data, please contact admin, {e}")
            self.close()

    def prepare_train_data(self, text8_file_s3_key):

        self.s3_train_data = f"s3://{self.output_bucket}/{text8_file_s3_key}"
        train_data = sagemaker.session.s3_input(
            self.s3_train_data,
            distribution="FullyReplicated",
            content_type="text/plain",
            s3_data_type="S3Prefix",
        )
        self.data_channels = {"train": train_data}

    def train(self):
        """
        train model with sagemaker BlazingText
        """
        self.bt_model = sagemaker.estimator.Estimator(
            self.container,
            self.role,
            train_instance_count=1,
            train_instance_type=config.TRAINING_INSTANCE_TYPE,
            train_volume_size=5,
            train_max_run=360000,
            input_mode="File",
            output_path=self.s3_output_location,
            sagemaker_session=self.session
        )
        
        self.bt_model.set_hyperparameters(
            mode= 'skipgram',
            epochs=5,
            min_count=5,
            sampling_threshold=0.0001,
            learning_rate=0.05,
            window_size=5,
            vector_dim=100,
            negative_samples=5,
            batch_size=11,  #  = (2*window_size + 1) (Preferred. Used only if mode is batch_skipgram)
            evaluation=True,  # Perform similarity evaluation on WS-353 dataset at the end of training
            subwords= True,
            min_n=3,
            max_n=6
        )
        try:
            self.bt_model.fit(inputs=self.data_channels, logs=True)
        except Exception as e:
            print("Failed to train model, please contact admin.")
            self.close()

    def download_trained_model(self, local_model_file="../output/model.tar.gz"):
        """
        download trained model from s3
        """
        try:
            self.s3_resource = boto3.resource("s3")
            key = self.bt_model.model_data[self.bt_model.model_data.find("/", 5) + 1 :]
            print(key)
            self.s3_resource.Bucket(self.output_bucket).download_file(key, local_model_file)
        except Exception as e:
            print("Failed to download trained model, please contact admin.")
            self.close()

    def evaluate_learned_model_vacs(self, local_model_file = "../output/model.tar.gz", to_local_path="../output/model"):
        """
        evaluate the learned model vectors
        """
        try:
            untar_file(local_model_file, to_local_path)
            pretty_print_json(to_local_path + "/eval.json")
            output_image_path = str.replace(to_local_path, "output/model", "temp") + "/model_vecs_" + self.data_time + ".png"
            plot_word_vecs_tsne(to_local_path + "/vectors.txt", None, output_image_path)
        except Exception as e:
            print("Failed to evaluate learned model vectors, please contact admin.")
            self.close()

    def deploy_trained_model(self, endpoint_name, trained_model_key):
        """
        deploy trained model to sagemaker endpoint
        """
        self.endpoint_name = config.ENDPOINT_NAME if endpoint_name is None else endpoint_name
        if trained_model_key:
            # check if the model exists
            if not self.S3Bucket.file_exists_on_s3(trained_model_key):
                print("Model does not exist, please contact admin.")
                self.close()
            else:
                model_location = f"s3://{self.output_bucket}/{trained_model_key}"
                self.bt_model = sagemaker.Model(model_data=model_location, # .tar.gz model S3 location
                    image_uri=self.container, # BlazingText docker image
                    role=self.role,
                    sagemaker_session=self.session)
                print( self.bt_model )
        try:   
            self.bt_endpoint = self.bt_model.deploy(
                    initial_instance_count=1,
                    instance_type=config.HOSTING_INSTANCE_TYPE,
                    endpoint_name=self.endpoint_name,
                )
                    
        except Exception as e:
            print("Failed to deploy model vectors, please contact admin.")
            print(e)
            self.close()

    def test_trained_model(self, word_list, endpoint_name):
        """
        evaluate trained model with word list
        """
        payload = {"instances": word_list}
        try:
            response = None
            if not endpoint_name and self.bt_endpoint:
                response = self.bt_endpoint.predict(
                    json.dumps(payload),
                    initial_args={"ContentType": "application/json", "Accept": "application/json"},
                )
                
            else:
                predictor = sagemaker.Predictor(
                    endpoint_name=endpoint_name, 
                    sagemaker_session=self.session,
                    serializer=sagemaker.serializers.JSONSerializer(),
                    #deserializer=sagemaker.deserializers.JSONDeserializer()
                )
                predictor.content_type = "application/json"
                predictor.accept = "application/json"
                response = predictor.predict(payload)

            vecs = json.loads(response)
            print(vecs)
            # output_image_path = "../temp/vecs_test_" + self.data_time + ".png"
            # plot_word_vecs_tsne(None, vecs, output_image_path)
        except Exception as e:
            print(e)
            self.close(self.endpoint_name )

    def close(self, endpoint_name=None):
        """
        delete endpoint and close s3_client, s3_resource, session
        """
        if endpoint_name:
            delete_endpoint(endpoint_name)
        if self.bt_model:
            self.bt_model = None
        if self.S3Bucket:
            self.S3Bucket = None
        if self.session:
            self.session = None

import os, sys
import glob
import sagemaker
import boto3
import json
import yaml

import sagemaker.deserializers
import sagemaker.serializers
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from bento.common.s3 import S3Bucket
from common.sagemaker_utils import get_sagemaker_session, delete_endpoint, get_sagemaker_execute_role, create_local_output_dirs
import common.sagemaker_config as config
from common.utils import untar_file, pretty_print_json, get_date_time, preprocess_text
from common.visualize_word_vecs import plot_word_vecs_tsne
from common.transform_json import transform_json_to_training_data
import numpy as np

class SemanticAnalysis:
    def __init__(self):
        """
        constructor of SemanticAnalysis
        """
        create_local_output_dirs(["../output", "../output/model", "../data", "../data/raw", "../data/raw/json","../temp"]) # create local directories if not exist.
        self.endpoint_name = None
        self.S3Bucket = S3Bucket(config.CRDCDH_S3_BUCKET)
        self.timestamp = get_date_time()
        self.data_time = get_date_time("%Y-%m-%d-%H-%M-%S")
        self.session = get_sagemaker_session(config.AWS_SAGEMAKER_USER, config.RUN_LOCAL)
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
        self.trained_model_vectors_path = "../output/model/vectors.txt"
        self.trained_model_vectors = None

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
            self.close(1)

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
                    self.close(1)
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
            self.close(1)

    def prepare_train_data(self, text8_file_s3_key, text8_file_local_path):

        if not text8_file_s3_key or not self.S3Bucket.file_exists_on_s3(text8_file_s3_key):
            if not text8_file_local_path or not os.path.exists(text8_file_local_path):
                print("No training data found on s3 bucket or local folder")
                self.close(1)
            else:
                # if text8 file in local folder, upload to designated s3 bucket
                text8_file_s3_key = os.path.join(config.TRAIN_DATA_PREFIX, f"{self.image_name}-{self.timestamp}/train_data_text8")
                print(f"Upload training data to s3 bucket, {text8_file_s3_key}.")
                try: 
                    self.S3Bucket.upload_file(text8_file_s3_key, text8_file_local_path)
                    print(f"Training data uploaded to s3 bucket, {text8_file_s3_key}.")
                except Exception as e:
                    print(f"Failed to upload training data, please contact admin, {e}")
                    self.close(1)

        self.s3_train_data = f"s3://{self.output_bucket}/{text8_file_s3_key}"
        train_data = sagemaker.session.s3_input(
            self.s3_train_data,
            distribution="FullyReplicated",
            content_type="text/plain",
            s3_data_type="S3Prefix",
        )
        self.data_channels = {"train": train_data}

    def train(self, algorithm):
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
        if algorithm == "FastText":
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
        elif algorithm == "Word2Vec":
            self.bt_model.set_hyperparameters(
                mode="batch_skipgram",
                epochs=5,
                min_count=5,
                sampling_threshold=0.0001,
                learning_rate=0.05,
                window_size=5,
                vector_dim=100,
                negative_samples=5,
                batch_size=11,  #  = (2*window_size + 1) (Preferred. Used only if mode is batch_skipgram)
                evaluation=True,  # Perform similarity evaluation on WS-353 dataset at the end of training
                subwords=False
            ) 
        elif algorithm == "TextClassification":
            self.bt_model.set_hyperparameters(
                mode="supervised",
                vector_dim=10, # Although a default of 100 is used for word2vec,10 is sufficient for text classification in most of the cases
                epochs=5,
                early_stopping=True,
                patience=4,       # Number of epochs to wait before early stopping if no progress on the validation set is observed
                min_epochs=5,
                learning_rate=0.05,
                buckets=2000000, # No. of hash buckets to use for word n-grams
                word_ngrams=2     # Number of word n-grams features to use.\
              )
        else:
            print("Invalid algorithm, please contact admin.")
            self.close(1)
        try:
            self.bt_model.fit(inputs=self.data_channels, logs=True)
        except Exception as e:
            print("Failed to train model, please contact admin.")
            self.close(1)

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
            self.close(1)

    def evaluate_learned_model_vacs(self, local_model_file = "../output/model.tar.gz", to_local_path="../output/model"):
        """
        evaluate the learned model vectors
        """
        try:
            untar_file(local_model_file, to_local_path)
            # pretty_print_json(to_local_path + "/eval.json")
            output_image_path = str.replace(to_local_path, "output/model", "temp") + "/model_vecs_" + self.data_time + ".png"
            plot_word_vecs_tsne(to_local_path + "/vectors.txt", None, output_image_path)
        except Exception as e:
            print("Failed to evaluate learned model vectors, please contact admin.")
            self.close(1)

    def deploy_trained_model(self, endpoint_name, trained_model_key):
        """
        deploy trained model to sagemaker endpoint
        """
        self.endpoint_name = config.ENDPOINT_NAME if endpoint_name is None else endpoint_name
        if trained_model_key:
            # check if the model exists
            if not self.S3Bucket.file_exists_on_s3(trained_model_key):
                print("Model does not exist, please contact admin.")
                self.close(1)
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
            self.close(1)

    def test_trained_model(self, word_list, endpoint_name):
        """
        evaluate trained model with word list
        """
        payload = {"instances": word_list}
        response = None
        try:
            
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
            similarity = cosine_similarity([vecs[0]["vector"]], [vecs[1]["vector"]])[0][0]
            print(similarity)
            # output_image_path = "../temp/vecs_test_" + self.data_time + ".png"
            # plot_word_vecs_tsne(None, vecs, output_image_path)
        except Exception as e:
            print(e)
            self.close(1)


    def evaluate_trained_model(self, endpoint_name, yaml_file):
        """
        evaluate trained model with test datasets
        """
        response = None
        eval_dict = yaml.load(open(yaml_file, 'r'), Loader=yaml.SafeLoader)
        eval_values = eval_dict.values()
        paired_words = []
        for dict in eval_values:
            paired_words.extend(dict.items())

        # Prepare data for inference
        print(f"Non-permissove/permissive pairs: {paired_words[:10]}")
        print("Total paired words: " + str(len(paired_words)))
        similarities = []
        try:
            
            if not endpoint_name and self.bt_endpoint:
                
                for key, val in paired_words:
                    if not key or not val:
                        continue
                    payload = {"instances": [preprocess_text(key), preprocess_text(val)]}
                    response = self.bt_endpoint.predict(
                        json.dumps(payload),
                        initial_args={"ContentType": "application/json", "Accept": "application/json"},
                    )
                    vecs = json.loads(response)
                    similarity = cosine_similarity([vecs[0]["vector"]], [vecs[1]["vector"]])[0][0]
                    similarities.append(similarity)
                
            else:
                predictor = sagemaker.Predictor(
                    endpoint_name=endpoint_name,
                    sagemaker_session=self.session,
                    serializer=sagemaker.serializers.JSONSerializer(),
                    # deserializer=sagemaker.deserializers.JSONDeserializer()
                )
                predictor.content_type = "application/json"
                predictor.accept = "application/json"
                for key, val in paired_words:
                    if not key or not val:
                        continue
                    payload = {"instances": [preprocess_text(key), preprocess_text(val)]}
                    # print(f"key: {key}, val: {val}")
                    response = predictor.predict(payload)
                    vecs = json.loads(response)
                    similarity = cosine_similarity([vecs[0]["vector"]], [vecs[1]["vector"]])[0][0]
                    similarities.append(similarity)
            print(similarities[:10])
            # Compute accuracy
            mean_similarity = np.mean(similarities)
            print(f"Mean similarity: {mean_similarity:.4f}")
            return mean_similarity
        except Exception as e:
            print(e)
            self.close(1)

    def search_similar_words(self, word, endpoint_name, top_k=5):
        """
        search similar words for a given word
        """
        response = None
        query_word = preprocess_text(word)
        try:
            if not endpoint_name and self.bt_endpoint:
                response = self.bt_endpoint.predict(
                    json.dumps({"instances": [query_word]}),
                    initial_args={"ContentType": "application/json", "Accept": "application/json"},
                )
            else:
                predictor = sagemaker.Predictor(
                    endpoint_name=endpoint_name,
                    sagemaker_session=self.session,
                    serializer=sagemaker.serializers.JSONSerializer(),
                    # deserializer=sagemaker.deserializers.JSONDeserializer()
                )
                predictor.content_type = "application/json"
                predictor.accept = "application/json"
                response = predictor.predict({"instances": [preprocess_text(word)]})

            query_vec = json.loads(response)[0]["vector"]
            print(f"{query_word}:{query_vec}")
            # Get the top_k most similar words to the given word
            if not self.trained_model_vectors:
                self.trained_model_vectors = load_model_vectors(self.trained_model_vectors_path) 
            similarities = {}
            for word, vec in self.trained_model_vectors.items():
                # print(word, vec)
                similarity = cosine_similarity([query_vec], [vec])[0][0]
                if query_word == word:
                    print(f"Similarity between {word} and {query_word}: {similarity:.4f}")
                    print(vec)
                similarities[word] = similarity
            # Sort words by similarity and get top_k words
            sorted_similar_words = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_k]
            print(sorted_similar_words)
            return sorted_similar_words
        except Exception as e:
            print(e)
            self.close(1)
    
    def search_from_permissive_values(self, word, permissive_values, endpoint_name, top_k=5):
        """
        search similar words for a given word from permissive values
        """
        response = None
        query_word = preprocess_text(word)
        query_words= [query_word]
        from_words = [preprocess_text(word) for word in permissive_values]
        query_words= [*query_words, *from_words]
        # print(query_words)
        word_vec = None
        similar_word = []
        try:
            if not endpoint_name and self.bt_endpoint:
                response = self.bt_endpoint.predict(
                    json.dumps({"instances": query_words}),
                    initial_args={"ContentType": "application/json", "Accept": "application/json"},
                )

            else:
                predictor = sagemaker.Predictor(
                    endpoint_name=endpoint_name,
                    sagemaker_session=self.session,
                    serializer=sagemaker.serializers.JSONSerializer(),
                    # deserializer=sagemaker.deserializers.JSONDeserializer()
                )
                predictor.content_type = "application/json"
                predictor.accept = "application/json"
                response = predictor.predict({"instances": query_words})
            word_vec = json.loads(response)
            # print(word_vec)
            query_vec = word_vec[0]["vector"]
            permissive_vectors = word_vec[1:]
            permissive_val_index = 0
            similarities = {}
            for item in permissive_vectors:
                print(item["word"], item["vector"])
                similarity = cosine_similarity([query_vec], [item["vector"]])[0][0]
                if query_word == item["word"]:
                    print(f"Similarity between {word} and {query_word}: {similarity:.4f}")
                    print(item["vector"])
                    similar_word.append(permissive_values[permissive_val_index])
                    return similar_word
                else:
                    similarities[item["word"]] = similarity
                permissive_val_index += 1
            # Sort words by similarity and get top_k words
            similar_word = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_k]
            print(similar_word)
            return similar_word
        except Exception as e:
            print(e)
            self.close(1)

    def close(self, exit_code, endpoint_name=None):
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
        try:
            sys.exit(exit_code)
        except SystemExit:
            msg = "Exited!" if exit_code == 1 else "Completed!"
            print(msg)

def load_model_vectors(model_vectors_file):
        """
        load model vectors from file
        """
        word_to_vec = {}
        try:
            with open(model_vectors_file, 'r') as file:
                i=0
                for line in file:
                    # skip line 1
                    if i==0:
                        i+=1
                        continue
                    parts = line.strip().split()
                    word = parts[0]
                    vector = list(np.float_(parts[1:]))
                    word_to_vec[word] = vector
            return word_to_vec
        except Exception as e:
            print("Failed to load model vectors, please contact admin.")
            raise e

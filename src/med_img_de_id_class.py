import os, sys
from datetime import datetime
import glob
import sagemaker
import json
import yaml, boto3, io, cv2
from botocore.exceptions import ClientError
from common.s3 import S3Bucket
from common.utils import untar_file, process_dict_tags, get_date_time, dict2yaml, get_boto3_session, dump_dict_to_json, yaml2dict
import numpy as np
from common.sagemaker_utils import get_sagemaker_session, delete_endpoint, get_sagemaker_execute_role, create_local_output_dirs, get_local_session
from imageio import imread
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib as mpl
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from common.de_id_utils import get_pii_boxes_nlp, generate_clean_image

class ProcessMedImage:
    def __init__(self, rule_config_file_path= '../configs/de-id/output.yaml'):
        """
        constructor of ProcessMedImage
        """
        create_local_output_dirs(["../output", "../output/model", "../data", "../data/raw", "../data/raw/json","../temp"]) # create local directories if not exist.
        self.endpoint_name = None
        self.boto3_session = get_boto3_session()
        self.timestamp = get_date_time()
        self.data_time = get_date_time("%Y-%m-%d-%H-%M-%S")
        self.role = get_sagemaker_execute_role(self.boto3_session)
        # print(self.role)  # This is the role that SageMaker would use to leverage AWS resources (S3, CloudWatch) on your behalf
        self.region_name = self.boto3_session.region_name
        self.s3_client = self.boto3_session.client('s3')
        self.rekognition= None
        self.comprehend = None
        self.comprehend_medical = None
        #Define the S3 bucket and object for the medical image we want to analyze.  Also define the color used for redaction.
        self.phi_detection_threshold = 0.00
        self.local_img_folder = "../images/med_phi_img/"
        self.rule_config_file_path = rule_config_file_path
        self.dicom_tags = None
        self.phi_tags = None
        self.sensitive_words = None
        self.set_rules(rule_config_file_path)

    def set_rules(self, rule_config_file_path):
        self.rules = yaml2dict(rule_config_file_path)["rules"]
        self.dicom_tags = self.rules['dicom_tags']
        self.phi_tags = [ tuple(item["tag"]) for item in self.rules['dicom_tags']]
        print(self.phi_tags)
        self.sensitive_words = self.rules['keywords']
        print(self.sensitive_words)


    def upload_dicom_file(self, src_bucket, src_key, local_dicom_path):
        """
        download dicom file
        """
        # Initialize S3 client
        self.s3_client .upload_file(local_dicom_path, src_bucket, src_key)
        print("DICOM file has been uploaded to the source a3 bucket.")

        # Load the DICOM data
        dicom_dataset = pydicom.dcmread(local_dicom_path)

        # Extract pixel array and convert to uint8 (if necessary)
        image_data = dicom_dataset.pixel_array

        if image_data.dtype != np.uint8:
            image_data = (image_data / np.max(image_data) * 255).astype(np.uint8)

        # Convert the pixel array to a PIL Image
        image = Image.fromarray(image_data)

        # Save the image to a bytes buffer in PNG format
        image_buffer = io.BytesIO()
        image.save(image_buffer, format='PNG')
        image_buffer.seek(0)

        # Upload the image to the destination S3 bucket
        key_png = src_key.replace('.dcm', '.png')
        self.s3_client.put_object(Bucket=src_bucket, Key=key_png, Body=image_buffer, ContentType='image/png')
        print("DICOM pixel data has be saved to the source s3 bucket as png for de-identification.")
        return True, dicom_dataset, image_data
    
    def detect_id_in_img(self, bucket, key, use_AI= False):
        """
        detect phi
        """
       # Step 3: Use Rekognition to detect text
        self.rekognition = self.boto3_session.client('rekognition')
        response = self.rekognition.detect_text(
            Image={
                'S3Object': {       
                    'Bucket': bucket,
                    'Name': key
                }
            }
        )
        all_ids = []
        if use_AI:
            detected_texts = [text for text in response['TextDetections']]
            for text in detected_texts:
                ids = self.detect_id_in_text_AI(text, is_image=True)
                if ids and len(ids) > 0:
                    all_ids.extend(ids)
        else:
            all_ids = get_pii_boxes_nlp(response)
            return all_ids

    def detect_id_in_text_AI(self, detected_text, is_image = False):
        # Step 4: Analyze text blocks for PHI using Comprehend Medical
        ids = []
        if not detected_text: return ids
        text = detected_text['DetectedText'] if is_image else detected_text
             
        # pii_entities = self.analyze_text_for_pii(text)
        # if pii_entities and len(pii_entities) > 0:
        #     print(f"PII detected in text: '{text}'")
        #     for entity in pii_entities:
        #         print(entity)
        #         if is_image: 
        #             print(text) 
        #             box = detected_text['Geometry']['BoundingBox']
        #             ids.append({"Text": text, "Text Block": box})
        #         else:
        #             ids.append(text)
        # else:
        phi_entities = self.analyze_text_for_phi(text)
        if phi_entities and len(phi_entities):
            print(f"PHI detected in text: '{text}'")
            for entity in phi_entities:
                print(f"Entity: {entity['Text']} - Type: {entity['Type']} - Confidence: {entity['Score']}")
                if is_image:  
                    box = detected_text['Geometry']['BoundingBox']
                    ids.append({"Text": text, "Text Block": box})
                else:
                    ids.append(text)
        return ids
    
    def analyze_text_for_pii(self, text):
        response = self.comprehend.detect_pii_entities(Text=str(text), LanguageCode='en')
        return response['Entities']

    def analyze_text_for_phi(self, text):
        response = self.comprehend_medical.detect_phi(Text=str(text))
        return response['Entities']
    
    def draw_img(self, img, dpi = 72):
        #Set the image color map to grayscale, turn off axis graphing, and display the image
        plt.rcParams["figure.figsize"] = [16,9]
        # Display the image
        plt.imshow(img, cmap=plt.cm.gray)
        plt.title('DICOM Image')
        plt.axis('off')  # Turn off the axis
        plt.show()

    def detect_id_in_tags(self, dicom_data):
        # Read the DICOM file
        # Print all the tags and their values
        tags = []
        all_ids = []
        ids_list = []
        ids = []
        self.comprehend= self.boto3_session.client('comprehend')
        self.comprehend_medical = self.boto3_session.client('comprehendmedical')
        for elem in dicom_data:
            tags.append(elem.tag)
            # Detect PHI/PII in the text
            # print(f"Tag: {elem.tag} - Name: {elem.name} - Value: {elem.value}")
            if not elem.value: pass
            if isinstance(elem.value, str):
                ids = self.detect_id_in_text_AI(elem.value, is_image=False)
                if ids and len(ids) > 0:
                    print(f"Tag: {elem}")
                    all_ids.extend(ids)
                    tuple = (elem.tag.group, elem.tag.element)
                    print(f"tag: {tuple}, Name:  {elem.name}")
                    self.dicom_tags.append({"tag": tuple, "name": elem.name})
            elif isinstance(elem.value, list):
                for item in elem.value:
                    if not item: pass
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if not value: pass
                            ids = self.detect_id_in_text_AI(value, is_image=False)
                            ids_list.extend(ids)
                            if ids and len(ids) > 0:
                                print(f"Tag: {elem}")
                    else:
                        ids = self.detect_id_in_text_AI(item, is_image=False)
                        ids_list.extend(ids)
                        if ids and len(ids) > 0:
                            print(f"Tag: {elem}")
                if len(ids_list) > 0:
                    print(f"Tag: {elem}")
                    tuple = (elem.tag.group, elem.tag.element)
                    print(f"tag: {tuple}, Name:  {elem.name}")
                    self.dicom_tags.append({"tag": tuple, "name": elem.name})
                    
        return tags, all_ids
    
    def convert_back_dicom(self, redacted_img_path, ds, local_de_id_dicom_path):
       # Load the PNG image
        image = Image.open(redacted_img_path)
        image_array = np.array(image)


        # Set the transfer syntax
        # ds.is_little_endian = True
        # ds.is_implicit_VR = False

        # Add the image data to the dataset
        ds.Rows, ds.Columns = image_array.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelData = image_array.tobytes()

        # Set the image creation date and time
        # dt = datetime.now()
        # ds.ContentDate = dt.strftime('%Y%m%d')
        # ds.ContentTime = dt.strftime('%H%M%S.%f')

        # Save the DICOM file
        ds.save_as(local_de_id_dicom_path)

        print(f"DICOM file has been created and saved to {local_de_id_dicom_path}.")


    def generate_anonymous_value(self, tag):
        # """Generate an anonymous value based on the tag."""
        # if tag == (0x0020, 0x000D):  # Study Instance UID
        #     return generate_uid()
        # elif tag == (0x0020, 0x000E):  # Series Instance UID
        #     return generate_uid()
        # elif tag == (0x0008, 0x0018):  # SOP Instance UID
        #     return generate_uid()
        # elif tag == (0x0020, 0x0052):  # Frame of Reference UID
        #     return generate_uid()
        # elif tag == (0x0020, 0x0200):  # Synchronization Frame of Reference UID
        #     return generate_uid()
        # else:
        return None

    def redact_tag_value(self, name, value, tag):
        """Function to replace sensitive data with placeholders or anonymous values."""
        if tag == (0x0010, 0x0010):  # Patient's Name
            return 'The^Patient'
        elif tag == (0x0008, 0x1060) or tag == (0x0008, 0x0090):  # physician's name
            return 'Dr.^Physician'
        elif tag == (0x0008, 0x1070):  # Operator's Name
            return 'Mr.^Operator'
        # elif isinstance(value, bool):
        #     return None
        elif isinstance(value, str):
            return None if not "UID" in name else self.generate_anonymous_value(tag)
        # elif isinstance(value, (int, float)):
        #     return None
        # elif isinstance(value, (datetime)):
        #     return None
        elif isinstance(value, list):
            return [None for _ in value]
        else:
            return None
        return value

    def de_identify_dicom(self, ds):
        # Redact PHI in the DICOM dataset
        redacted_value = None
        #  redact date and UID tags
        detected_tags = []
        for item in ds:
            name = item.name
            for key in self.sensitive_words:
                if key in name:
                    redacted_value = self.redact_tag_value(name, item.value, item.tag)
                    # print(f"Tag: {item.tag} - {name} - Value: {item.value} - Redacted Value: {redacted_value}")
                    print(f"Tag: {item} - Redacted Value: {redacted_value}")
                    item.value = redacted_value
                    detected_tags.append(item.tag)
        for tag in self.phi_tags:
            if tag in ds and not tag in detected_tags:
                name = ds[tag].name
                ds_tag = ds[tag]
                redacted_value = self.redact_tag_value(name, ds[tag].value, ds_tag)
                print(f"Tag: {ds_tag} - Redacted Value: {redacted_value}")
                ds[tag].value = redacted_value

        print(f"Redacted DICOM matadata")

    def redact_id_in_image(self, bucket_name, img_key, all_ids, output_image_path):
        """
        redact id in image
        """
        # Download image object from S3
        response = self.s3_client.get_object(Bucket=bucket_name, Key=img_key)
        image_data = response['Body'].read()

        # Convert image data to numpy array
        np_arr = np.frombuffer(image_data, np.uint8)
        
        # Decode image data to OpenCV format
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        return generate_clean_image(img, [ item["Text Block"]  for item in all_ids], output_image_path)

    def get_s3_bucket(self):
        """
        get s3 bucket
        """
        return self.S3Bucket
    
    def update_rules_in_configs(self, tags, keywords, config_file = None):
        """Convert dictionary to YAML and save to file."""
        # Process the dictionary
        tags_key = "dicom_tags"
        if not config_file: 
            config_file = self.rule_config_file_path
        dictionary = {}
        dictionary["rules"] = self.rules
        dictionary["rules"][tags_key] = tags
        dictionary["rules"]["keywords"] = keywords
        dictionary["rules"][tags_key] = process_dict_tags( dictionary["rules"][tags_key], "tag")
        print(dictionary)

        # save change to the rules config file.
        dict2yaml(dictionary, config_file)

        
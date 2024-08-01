import os, sys
from datetime import datetime
import glob
import sagemaker
import json
import yaml, boto3, io, cv2
from botocore.exceptions import ClientError
from common.s3 import S3Bucket
from common.utils import untar_file, pretty_print_json, get_date_time, preprocess_text, get_boto3_session, dump_dict_to_json
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

phi_tags = [
            (0x0010, 0x0010),  # Patient's Name
            (0x0010, 0x0020),  # Patient ID
            (0x0010, 0x0030),  # Patient's Birth Date
            (0x0010, 0x0040),  # Patient's Sex
            (0x0010, 0x1000),  # Other Patient IDs
            (0x0010, 0x1001),  # Other Patient Names
            (0x0010, 0x1010),  # Patient's Age
            (0x0010, 0x1020),  # Patient's Size
            (0x0010, 0x1030),  # Patient's Weight
            (0x0010, 0x1090),  # Medical Record Locator
            (0x0020, 0x000D),  # Study Instance UID
            (0x0020, 0x0010),  # Study ID
            (0x0008, 0x0020),  # Study Date
            (0x0008, 0x0030),  # Study Time
            (0x0008, 0x0090),  # Referring Physician's Name
            (0x0008, 0x0050),  # Accession Number
            (0x0008, 0x1030),  # Study Description
            (0x0008, 0x1048),  # Physician(s) of Record
            (0x0008, 0x1060),  # Name of Physician(s) Reading Study
            (0x0020, 0x000E),  # Series Instance UID
            (0x0020, 0x0011),  # Series Number
            (0x0008, 0x0021),  # Series Date
            (0x0008, 0x0031),  # Series Time
            (0x0008, 0x103E),  # Series Description
            (0x0008, 0x1070),  # Operator's Name
            (0x0008, 0x0070),  # Manufacturer
            (0x0008, 0x0080),  # Institution Name
            (0x0008, 0x1010),  # Station Name
            (0x0008, 0x1040),  # Institutional Department Name
            (0x0008, 0x1090),  # Manufacturer's Model Name
            (0x0018, 0x1000),  # Device Serial Number
            (0x0018, 0x1016),  # Secondary Capture Device Manufacturer
            (0x0018, 0x1018),  # Secondary Capture Device Manufacturer's Model Name
            (0x0008, 0x0018),  # SOP Instance UID
            (0x0020, 0x0052),  # Frame of Reference UID
            (0x0020, 0x0200),  # Synchronization Frame of Reference UID
            (0x0008, 0x0081),  # Institution Address
            (0x0038, 0x0010),  # Admission ID
            (0x0038, 0x0061),  # Discharge Diagnosis Description
            (0x0038, 0x0300),  # Current Patient Location
            (0x0040, 0x0001),  # Scheduled Station AE Title
            (0x0040, 0x0002),  # Scheduled Procedure Step Start Date
            (0x0040, 0x0003),  # Scheduled Procedure Step Start Time
            (0x0040, 0x0006),  # Scheduled Performing Physician's Name
            (0x0040, 0x0010),  # Scheduled Station Name
            (0x0040, 0x1001),  # Requested Procedure ID
            (0x0040, 0x1002),  # Reason for the Requested Procedure
            (0x0040, 0x2001),  # Reason for the Imaging Service Request
            (0x0008, 0x1084),  # Admitting Diagnoses Description
            (0x0008, 0x1120),  # Referenced Patient Sequence
            (0x0010, 0x21B0),  # Additional Patient History
            (0x0038, 0x0040),  # Discharge Diagnosis Description
            (0x0040, 0x0241),  # Performed Station AE Title
            (0x0040, 0x0242),  # Performed Station Name
            (0x0040, 0x0243),  # Performed Location
            (0x0040, 0x0244),  # Performed Procedure Step Start Date
            (0x0040, 0x0245),  # Performed Procedure Step Start Time
            (0x0040, 0x0253),  # Performed Procedure Step ID
            (0x0008, 0x002a),  # Acquisition DateTime
            (0x0008, 0x0022),  # Acquisition Date
            (0x0008, 0x0023),  # Content Date
            (0x0008, 0x0033),  # Content Time 
            (0x0008, 0x0024),  # Overlay Date
            (0x0008, 0x0025),  # Curve Date 
            (0x0008, 0x0026),  # Overlay Time
            (0x0008, 0x0027),  # Curve Time
            (0x0008, 0x0028),  # Acquisition Time
            (0x0008, 0x0029),  # Acquisition Duration
            (0x0008, 0x002b),  # Acquisition Number
            (0x0008, 0x0016),  # SOP Class UID
            (0x0088, 0x0140),  # Storage Media File-set UID
            (0x0008, 0x0094),  # Referring Physician's Telephone Numbers
            (0x0013, 0x1013),  # Name: Private tag data
        ]

class ProcessMedImage:
    def __init__(self):
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
            detected_texts = [text['DetectedText'] for text in response['TextDetections']]
            for text in detected_texts:
                ids = self.detect_id_in_text_AI(text, is_image=True)
                if ids and len(ids) > 0:
                    all_ids.extend(ids)
        else:
            all_ids = get_pii_boxes_nlp(response)
            return all_ids

    def detect_id_in_text_AI(self, text, is_image = False):
        # Step 4: Analyze text blocks for PHI using Comprehend Medical
        ids = []
        if not text: return ids
        pii_entities = self.analyze_text_for_pii(text)
        if pii_entities and len(pii_entities) > 0:
            print(f"PII detected in text: '{text}'")
            for entity in pii_entities:
                print(entity)
                if is_image:  
                    box = text['Geometry']['BoundingBox']
                    ids.append({"Text": text, "Text Block": box})
                else:
                    ids.append(text)
        else:
            phi_entities = self.analyze_text_for_phi(text)
            if phi_entities and len(phi_entities):
                print(f"PHI detected in text: '{text}'")
                for entity in phi_entities:
                    print(f"Entity: {entity['Text']} - Type: {entity['Type']} - Confidence: {entity['Score']}")
                    if is_image:  
                        box = text['Geometry']['BoundingBox']
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
        # Display the image
        plt.imshow(img, cmap=plt.cm.gray)
        plt.rcParams["figure.figsize"] = [16,9]
        plt.title('DICOM Image')
        plt.axis('off')  # Turn off the axis
        plt.show()

    def detect_id_in_tags(self, dicom_data):
        # Read the DICOM file
        # Print all the tags and their values
        tags = []
        all_ids = []
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
                    print(f"Tag: {elem.tag} - Name: {elem.name} - Value: {elem.value}")
                    all_ids.extend(ids)
            elif isinstance(elem.value, list):
                for item in elem.value:
                    if not item: pass
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if not value: pass
                            ids = self.detect_id_in_text_AI(value, is_image=False)
                            all_ids.extend(ids)
                            if ids and len(ids) > 0:
                                print(f"Tag: {elem.tag} - Name: {elem.name} - Value: {elem.value}")
                    else:
                        ids = self.detect_id_in_text_AI(item, is_image=False)
                        all_ids.extend(ids)
                        if ids and len(ids) > 0:
                            print(f"Tag: {elem.tag} - Name: {elem.name} - Value: {elem.value}")
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
        """Generate an anonymous value based on the tag."""
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
        elif isinstance(value, bool):
            return None
        elif isinstance(value, str):
            return self.generate_anonymous_value(tag) if 'UID' in name else None
        elif isinstance(value, (int, float)):
            return None
        elif isinstance(value, (datetime)):
            return None
        elif isinstance(value, list):
            return [None for _ in value]
        else:
            return None
        return value

    def de_identify_dicom(self, ds):
        # Redact PHI in the DICOM dataset
        redacted_value = None
        #  redact date and UID tags
        for item in ds:
            name = item.name
            if "Date" in name or "Time" in name or "UID" in name or "Address" in name or "Telephone" in name or "Fax" in name:
                redacted_value = None
                print(f"Tag: {item.tag} - {name} - Value: {item.value} - Redacted Value: {redacted_value}")
                item.value = redacted_value
        for tag in phi_tags:
            if tag in ds:
                name = ds[tag].name
                if "Date" in name or "Time" in name or "UID" in name or "Address" in name or " Numbers" in name: pass
                redacted_value = self.redact_tag_value(name, ds[tag].value, tag)
                print(f"Tag: {tag} - {ds[tag].name} - Value: {ds[tag].value} - Redacted Value: {redacted_value}")
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
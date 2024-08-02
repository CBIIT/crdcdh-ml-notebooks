import cv2
import spacy
import yaml

def generate_clean_image(image, boxes, output_image_path):
    # image = cv2.imread(image_path)
    height, width = image.shape

    for box in boxes:
        x1 = int(box['Left'] * width)
        y1 = int(box['Top'] * height)
        x2 = int((box['Left'] + box['Width']) * width)
        y2 = int((box['Top'] + box['Height']) * height)

        # Apply Gaussian blur to the detected region
        roi = image[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (23, 23), 30)
        image[y1:y2, x1:x2] = blurred_roi

    cv2.imwrite(output_image_path, image)
    cv2.destroyAllWindows()

    return True

def load_image(image_path):
    with open(image_path, 'rb') as image_file:
        return image_file.read()
    
def get_text_boxes(detected_txt):
    text_detections = detected_txt['TextDetections']
    boxes = []
    for text in text_detections:
        if text['Type'] == 'LINE':
            box = text['Geometry']['BoundingBox']
            boxes.append(box)
    return boxes

def get_pii_patterns():
    pii_patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        'dob_usa': r'\b\d{2}/\d{2}/\d{4}\b',  # Date (MM/DD/YYYY)
        'dob_euro': r'\b\d{2}-\d{2}-\d{4}\b',  # Date (DD-MM-YYYY)
        'dob_usa_short': r'\b\d{2}/\d{2}/\d{2}\b',  # Date (MM/DD/YY)
        'dob_euro_short': r'\b\d{2}-\d{2}-\d{2}\b',  # Date (DD-MM-YY)
        'phone': r'\b\d{3}-\d{7}\b',  # Phone Number
        'zip': r'\b\d{5}(?:[-\s]\d{4})?\b',  # Zip Code
        'driver_license': r'\b\d{3}-\d{2}-\d{4}\b',  # Driver's License
        'ssn_1': r'\b\d{9}\b',  # Social Security Number
        'insurance_policy': r'\b\d{3}-\d{2}-\d{4}\b',  # Insurance Policy Number
        'post_code': r'\b\d{5}(?:[-\s]\d{4})?\b',  # Postal Code
        'med_record_num': r'\b\d{3}-\d{2}-\d{4}\b',  # Medical Record Number
        'health_insurance_num': r'\b\d{3}-\d{2}-\d{4}\b',  # Health Insurance Card Number
        'bank_acct_num': r'\b\d{3}-\d{2}-\d{4}\b',  # Bank Account Number
        'routing_num': r'\b\d{3}-\d{2}-\d{4}\b',  # Routing Number
        'credit_card_num': r'\b\d{3}-\d{2}-\d{4}\b',  # Credit Card Number
    }
    return pii_patterns 

def is_pii_nlp(text):
    # Define custom entities for PII
    pii_entities = ['PERSON', 'GPE', 'ORG', 'LOC', 'DATE', 'TIME']
    # Load a pre-trained SpaCy model
    nlp = spacy.load('en_core_web_sm', disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    # print(text)
    doc = nlp(text.lower())
    for ent in doc.ents:
        # print(ent.text, ent.label_)
        if ent.label_ in pii_entities:
            return True
    return False

def get_pii_boxes_nlp(response):
    text_detections = response['TextDetections']
    pii_boxes = []
    for text in text_detections:
        if text['Type'] == 'LINE':
            detected_text = text['DetectedText']
            print("Found text: " + detected_text)
            result = is_pii_nlp(detected_text)
            if result:
                box = text['Geometry']['BoundingBox']
                pii_boxes.append({"Text": detected_text, "Text Block": box})
    return pii_boxes


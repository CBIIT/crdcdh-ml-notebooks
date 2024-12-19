from pymongo import MongoClient
import json
import requests
from bento.common.utils import get_logger, LOG_PREFIX, APP_NAME
import os

MONGO_URI = "mongodb://localhost:27017/"
MONGO_DATABASE = "local"
MONGO_COLLECTION = "CDE"
CDE_CODE = "CDECode"
CDE_FULLNAME = "CDEFullName"
CDE_URL = "https://cadsrapi.cancer.gov/rad/NCIAPI/1.0/api/DataElement/"
NCIT_URL = "https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/"
DATA_ELEMENT = "DataElement"
VALUE_DOMAIN = "ValueDomain"
PERMISSIBLE_VALUES = "PermissibleValues"
VALUE_MEANING = "ValueMeaning"
CONCEPTS = "Concepts"
SYNONYMS = "synonyms"

if LOG_PREFIX not in os.environ:
    os.environ[LOG_PREFIX] = 'CDE_NCIT'
    os.environ[APP_NAME] = 'CDE_NCIT'
log = get_logger('CDE_NCIT')

client = MongoClient(MONGO_URI)
db = client[MONGO_DATABASE]
collection = db[MONGO_COLLECTION]

# Query the collection (for example, find all documents)
documents = collection.find()

mongo_list = []
error_cde_list = []

# Iterate over each document
for document in documents:
    cde_code = document[CDE_CODE]
    prop = document[CDE_FULLNAME]
    if cde_code is None:
        continue
    log.info(f"The cde code for the property {prop} is {cde_code}")
    cde_url = CDE_URL + cde_code
    cde_header = {"accept": "application/json"}
    response = requests.get(cde_url, headers=cde_header)
    if response.status_code != 200:
        error_cde_list.append(cde_code)
        log.error(f"CDE Request failed with status code {response.status_code} and CDE code {cde_code}")
        continue
    data = json.loads(response.content)
    if data["status"] == "error":
        log.error(data["message"])
        continue
    pv_list = data.get(DATA_ELEMENT,{}).get(VALUE_DOMAIN,{}).get(PERMISSIBLE_VALUES)
    if pv_list is None:
        log.error("Can not find the Permissible Values")
        continue
    if len(pv_list) == 0:
        log.error("Can not find the Permissible Values")
        continue
    for pv in pv_list:
        value = pv.get("value")
        concept_list = pv.get(VALUE_MEANING,{}).get(CONCEPTS)
        if concept_list is None:
            log.error(f"Can not find the Concept values for the CDE value {value}")
            continue
        for concept in concept_list:
            ncit_code = concept.get("conceptCode")
            if ncit_code is None:
                log.error(f"Can not find the NCIT code for the CDE value {value}")
                continue
            ncit_url= NCIT_URL + ncit_code
            ncit_response = requests.get(ncit_url)
            if ncit_response.status_code != 200:
                log.error(f"NCIT Request failed with status code {ncit_response.status_code}")
                continue
            ncit_data_string = ncit_response.content.decode('utf-8')
            ncit_data = json.loads(ncit_data_string)
            synonyms_list = ncit_data.get(SYNONYMS)
            if synonyms_list is not None:
                for synonym in synonyms_list:
                    if synonym["name"] != value:
                        s = synonym["name"]
                        log.info(f'The synonym for the value {value} under the property {prop} is {s}')
                        mongo_list.append({"synonym_term": s, "equivalent_term": value})

if len(mongo_list) > 0:
    cleaned_mongo_list = [dict(t) for t in {tuple(d.items()) for d in mongo_list}]

with open('mongodb_ncit_cde.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_mongo_list, f, ensure_ascii=False, indent=4)
if len(error_cde_list) > 0:
    with open('error_cde_code.txt', 'w') as error_cde_file:
        for error_cde_code in error_cde_list:
            error_cde_file.write(f"{error_cde_code}\n")
    
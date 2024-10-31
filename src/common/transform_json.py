import glob
import boto3
import os
import json
import re
import requests
import yaml
from common.sagemaker_config import CRDCDH_S3_BUCKET
from nltk.corpus import stopwords
import nltk

def write_list_to_txt(input_list, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w') as file:
        for item in input_list:
            file.write(item + '\n')
    print(f"The training dataset is created and stored in {file_path}")

def search_evs_list(data, search_str):
    for item in data:
        if search_str in item.keys():
            return item[search_str]

def find_ncit_code(data, ncit_key):
    values = []
    
    def search_dict(d):
        if isinstance(d, dict):
            for key, value in d.items():
                if key == ncit_key:
                    values.append(value)
                elif isinstance(value, dict):
                    search_dict(value)
                elif isinstance(value, list):
                    for item in value:
                        search_dict(item)
    
    search_dict(data)
    return values

def extract_transform_model_cde_data(props):
    mongo_list = []
    for prop in props["PropDefinitions"].keys():
        try:
            terms = props["PropDefinitions"][prop].get("Term")
            prop_defination = props["PropDefinitions"][prop]["Desc"]
            mongo_list.append(f'{prop} is {prop_defination}')
            if terms is None:
                continue
            else:
                cde_code = terms[0].get("Code")
                if cde_code is None:
                    continue
                cde_url = "https://cadsrapi.cancer.gov/rad/NCIAPI/1.0/api/DataElement/" + cde_code
                cde_header = {"accept": "application/json"}
                response = requests.get(cde_url, headers=cde_header)
                if response.status_code != 200:
                    continue
                data = json.loads(response.content)
                data_element_definition = data.get("DataElement",{}).get("definition")
                if data_element_definition is None:
                    continue
                else:
                    mongo_list.append(f'{prop} is {data_element_definition}')
                ncit_codes = find_ncit_code(data, "conceptCode")
                if len(ncit_codes) > 0:
                    for ncit_code in ncit_codes:
                        ncit_url= "https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/" + ncit_code
                        ncit_response = requests.get(ncit_url)
                        if ncit_response.status_code != 200:
                            continue
                        else:
                            ncit_data_string = ncit_response.content.decode('utf-8')
                            ncit_data = json.loads(ncit_data_string)
                            synonyms_list = ncit_data.get("synonyms")
                            defin_list = ncit_data.get("definitions")
                            if defin_list is not None:
                                if synonyms_list is not None:
                                    for synonym in synonyms_list:
                                        s = synonym["name"]
                                        #
                                        for defin in defin_list:
                                            definition = defin['definition']
                                            mongo_list.append(f"{s} is {definition}")
        except Exception as e:
            print(f"{prop}:{e}")
    return mongo_list

def clean_training_data(input_list):
    updated_input_list = []
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    for item in input_list:
        text = re.sub(r'[^\w\s\']',' ', item)
        text = re.sub(r'[ \n]+', ' ', text)
        text = text.strip().lower()
        tokens = text.lower().split()
        filtered_tokens = [word for word in tokens if word not in stop_words]
        filter_string = ' '.join(filtered_tokens)
        updated_input_list.append(filter_string)
        #text = text+"\n"
    updated_input_list = list(set(updated_input_list))
    return updated_input_list

def extract_transform_gdc_data(gdc_props_data, gdc_values_data, ncit_data):
    matching_info = []
    code_value = "code_value"
    code_name = "code_name"
    n_mame = "nm"
    n_code = "n_c"
    definitions = "definitions"
    definition = "definition"
    synonyms = "synonyms"
    updated_matching_info = []
    defnition_list = []
    training_set = []
    for key, value in gdc_props_data.items():
        matching_info.append(
        {
            code_name: key.split(".")[len(key.split("."))-1],
            code_value: [value]
        }
        )
    for key in gdc_values_data.keys():
        for items in gdc_values_data[key]:
            if items[n_mame] != "":
                matching_info.append(
                {
                    code_name: items[n_mame].split(".")[len(items[n_mame].split("."))-1],
                    code_value: items[n_code]
                }
            )
    for i in matching_info:
        for v in i[code_value]:
            updated_matching_info.append(
                {
                code_name: i[code_name],
                code_value: v
            }
            )
    for i in updated_matching_info:
        if i[code_value] != "":
            ncit_dict = search_evs_list(ncit_data, i[code_value])
            defnition_item = i
            try:
                defnition_item[definitions] = ncit_dict[definitions]
            except Exception as e:
                defnition_item[definitions] = [{
                        definition: ""
                    }]
            try:
                defnition_item[synonyms] = ncit_dict[synonyms]
            except Exception as e:
                defnition_item[synonyms] = ""
            defnition_list.append(defnition_item)
    training_set = generate_training_data(defnition_list, training_set, definitions, definition, code_name, synonyms)
    training_set = list(set(training_set))
    return training_set
    
def extract_transform_evs_data(evsip_data, ncit_data):
    p_code = "p_n_code"
    v_code = "v_n_code"
    p_name = "p_name"
    v_name = "v_name"
    code_property = "code_property"
    code_value = "code_value"
    code_name = "code_name"
    definitions = "definitions"
    definition = "definition"
    synonyms = "synonyms"
    matching_info = []
    training_set = []
    defnition_list = []
    def search_evs_dict(parent_key, data, search_str):
        for key, value in data.items():
            if search_str in str(key):
                name = parent_key
                if key == p_code:
                    name = data[p_name]
                elif key == v_code:
                    name = data[v_name]
                matching_info.append(
                {
                    code_property: key,
                    code_value: value,
                    code_name: name
                }
                )
            if isinstance(value, dict):
                search_evs_dict(key, value, search_str)
            elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            search_evs_dict(key, item, search_str)
    search_evs_dict("", evsip_data, "code")
    for i in matching_info:
        if i[code_value] != "":
            ncit_dict = search_evs_list(ncit_data, i[code_value])
            defnition_item = i
            try:
                defnition_item[definitions] = ncit_dict[definitions]
            except Exception as e: #if code not in ncit data
                defnition_item[definitions] = [{
                        definition: ""
                    }]
            defnition_item[synonyms] = ncit_dict[synonyms]
            defnition_list.append(defnition_item)
    training_set = generate_training_data(defnition_list, training_set, definitions, definition, code_name, synonyms)
    training_set = list(set(training_set))
    return training_set

def generate_training_data(defnition_list, training_set, definitions, definition, code_name, synonyms):
    termName = "termName"
    for i in defnition_list:
        for defn in i[definitions]:
            if defn[definition] != "":
                training_set.append(i[code_name].lower() + " is " + defn[definition].lower())
            else:
                training_set.append(i[code_name].lower())
        for syn in i[synonyms]:
            for defn in i[definitions]:
                if defn[definition] != "":
                    training_set.append(syn[termName].lower() + " is " + defn[definition].lower())
                else:
                    training_set.append(syn[termName].lower())
    return training_set
            

def download_from_S3(s3, raw_data_folder, s3_json_file_prefix):
    response = s3.list_objects_v2(Bucket=CRDCDH_S3_BUCKET, Prefix=s3_json_file_prefix)
    if 'Contents' in response:
        if not os.path.exists(raw_data_folder):
            os.makedirs(raw_data_folder)
        for obj in response['Contents']:
            s3_file_key = obj['Key']
            if s3_file_key.endswith('.json') or s3_file_key.endswith('.yml') or s3_file_key.endswith('.yaml'):
                local_file_path = os.path.join(raw_data_folder, os.path.basename(s3_file_key))
                s3.download_file(CRDCDH_S3_BUCKET, s3_file_key, local_file_path)
                print(f'Downloaded {s3_file_key} to {local_file_path} successfully')


def transform_json_to_training_data(s3_json_file_prefix, raw_data_folder, s3_training_data_file_key, training_data_folder, session):
    NCIT = "ncit"
    GDC_PROPS = "gdc_props.json"
    GDC_VALUES = "gdc_values.json"
    s3= session.client('s3')
    try:
        download_from_S3(s3, raw_data_folder, s3_json_file_prefix)
        json_files = glob.glob('{}/*.json'.format(raw_data_folder))
        yaml_files = glob.glob('{}/*.yaml'.format(raw_data_folder))
        yml_files = glob.glob('{}/*.yml'.format(raw_data_folder))
        schema_files = yml_files + yaml_files
        ncit_files = [n for n in json_files if NCIT in n]
        ncit_file = ncit_files[0]
        with open(ncit_file, 'r') as file:
            ncit_data = json.load(file)
        training_data_list = []
        for json_file in json_files:
            if NCIT not in json_file and GDC_PROPS not in json_file and GDC_VALUES not in json_file:
                with open(json_file, 'r') as file:
                    evsip_data = json.load(file)
                evs_training_data = extract_transform_evs_data(evsip_data, ncit_data)
                if len(evs_training_data) > 0:
                    training_data_list.extend(evs_training_data)
            elif GDC_PROPS in json_file:
                with open(os.path.join(raw_data_folder, GDC_PROPS), 'r') as file:
                    gdc_props_data = json.load(file)
                with open(os.path.join(raw_data_folder, GDC_VALUES), 'r') as file:
                    gdc_values_data = json.load(file)
                
                gdc_training_data = extract_transform_gdc_data(gdc_props_data, gdc_values_data, ncit_data)
                if len(gdc_training_data) > 0:
                    training_data_list.extend(gdc_training_data)
        for schema_file in schema_files:
            model_props = yaml.load(open(schema_file, 'r'), Loader=yaml.SafeLoader)
            model_cde_training_data = extract_transform_model_cde_data(model_props)
            if len(model_cde_training_data) > 0:
                training_data_list.extend(model_cde_training_data)

        training_data_list = clean_training_data(training_data_list)
        output_file_path = os.path.join(training_data_folder, os.path.basename(s3_training_data_file_key))
        write_list_to_txt(training_data_list, output_file_path)
        s3.upload_file(output_file_path, CRDCDH_S3_BUCKET, s3_training_data_file_key)
        print(f'Uploaded {output_file_path} to {s3_training_data_file_key} successfully')
    except Exception as e:
        print(f'Failed to transform data in transformer: {e}')
        raise Exception(f'Failed to transform data!')
    finally:
        if s3:
            s3.close()
            s3 = None

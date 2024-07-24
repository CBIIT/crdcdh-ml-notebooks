import tarfile, json, re, requests, yaml, boto3, sys
from datetime import datetime 

def untar_file(tar_file, dest_dir):
    """
    untar file to dest_dir
    """
    with tarfile.open(tar_file, 'r:gz') as _tar:            
        for member in _tar:
            if member.isdir():
                continue
            _tar.makefile(member, dest_dir + '/' + member.name)

def load_json(json_file):
    """
    load json file
    """
    with open(json_file, 'r') as f:
        return json.load(f)


def pretty_print_json(json_file):
    """
    print json file prettily
    """
    print(json.dumps(load_json(json_file), indent=4))

def get_date_time(format = "%Y-%m-%d-%H-%M-%S-%f"):
    """
    get current time in format
    """  
    return datetime.strftime(datetime.now(), format)[:-3]

# Function to preprocess text
def preprocess_text(text):
    text = text.lower() if text else ''  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text) if text else ''  # Remove punctuation
    return text

def get_boto3_session(aws_profile=None):
    return boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()

"""
download file from url and load into dict 
:param: url string
:return: value dict
"""
def download_file_to_dict(url):
    # NOTE the stream=True parameter below
    file_ext = url.split('.')[-1]
    with requests.get(url) as r:
        if r.status_code > 400: 
            raise Exception(f"Can't find model file at {url}, {r.content}!")
        if file_ext == "json":
            return r.json()
        elif file_ext in ["yml", "yaml"]: 
            return yaml.safe_load(r.content)
        else:
            raise Exception(f'File type is not supported: {file_ext}!')
        
"""
Dump list of dictionary to json file, caller needs handle exception.
:param: dict_list as list of dictionary
:param: file_path as str
:return: boolean
"""
def dump_dict_to_json(dict, path):
    if not dict or len(dict.items()) == 0:
        return False 
    output_file = open(path, 'w', encoding='utf-8')
    json.dump(dict, output_file) 
    return True

"""
Extract exception type name and message
:return: str
"""
def get_exception_msg():
    ex_type, ex_value, exc_traceback = sys.exc_info()
    return f'{ex_type.__name__}: {ex_value}'


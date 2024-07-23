import os, re
from bento.common.utils import get_logger
from common.constants import MODELS_DEFINITION_FILE, LIST_DELIMITER_PROP
from common.utils import download_file_to_dict, get_exception_msg


YML_FILE_EXT = ["yml", "yaml"]
DEF_MODEL_FILE = "model-file"
DEF_MODEL_PROP_FILE = "prop-file"
DEF_VERSION = "versions"
MODE_ID_FIELDS = "id_fields"
DEF_SEMANTICS = "semantics"
DEF_FILE_NODES = "file-nodes"
PROP_DEFINITIONS = 'PropDefinitions'
PROP_TYPE = 'Type'
PROP_ENUM = 'Enum'
ITEM_TYPE = 'item_type'
VALUE_TYPE = 'value_type'
prop_list_types = [
    "value-list", # value_type: value type: list
            # a string with comma ',' characters as deliminator, e.g, "value1,value2,value3", represents a value list value1,value2,value3
    "list" # value_type: list
            # a string with asterisk '*' characters as deliminator, e.g, "value1*value2+value3", represents a array [value1, value2, value3]
]

class DataModelFactory:
    
    def __init__(self, model_def_loc, tier):
        self.log = get_logger('Models')
        self.models = None
        msg = None
        # get models definition file, content.json in models dir
        self.model_def_dir = os.path.join(model_def_loc, tier)
        models_def_file_path = os.path.join(self.model_def_dir, MODELS_DEFINITION_FILE)
        self.models_def = download_file_to_dict(models_def_file_path)
        # to do check if  self.models_def is a dict
        if not isinstance(self.models_def, dict):
            msg = f'Invalid models definition at "{models_def_file_path}"!'
            self.log.error(msg)
            raise Exception(msg)
        self.models = {}
        for k, v in self.models_def.items():
            data_common = k
            versions = v[DEF_VERSION]
            for version in versions:
                self.create_model(data_common, version)

    """
    create a data model dict by parsing yaml model files
    """
    def create_model(self, data_common, version):
        dc = data_common.upper()
        v = self.models_def[dc]
        model_dir = os.path.join(self.model_def_dir, os.path.join(dc, version))
        #process model files for the data common
        # file_name= os.path.join(model_dir, v[DEF_MODEL_FILE])
        props_file_name = os.path.join(model_dir, v[DEF_MODEL_PROP_FILE])
        #process model files for the data common
        try:
            result, properties_permissive_values, msg = self.parse_model_props(props_file_name)
            if not result:
                self.log.error(msg)
                return
            self.models[model_key(dc, version)] = properties_permissive_values
        except Exception as e:
            self.log.exception(e)
            msg = f"Failed to create data model: {data_common}/{version}!"
            self.log.exception(f"{msg} {get_exception_msg()}")


    """
    parse model property file
    """
    def parse_model_props(self, model_props_file):
        properties = None
        permissive_value_dic = {}
        values = None
        msg = None
        try:
            self.log.info('Reading propr file: {} ...'.format(model_props_file))
            if model_props_file and '.' in model_props_file and model_props_file.split('.')[-1].lower() in YML_FILE_EXT:
                properties = download_file_to_dict(model_props_file).get(PROP_DEFINITIONS)
                if not properties:
                    msg = f'Invalid model properties file: {model_props_file}!'
                    self.log.error(msg)
                    return False, None, msg
        except Exception as e:
            self.log.exception(e)
            msg = f'Failed to read yaml file to dict: {model_props_file}!'
            self.log.exception(msg)
            raise e
        
        # filter properties with enum and value list
        for prop_name, prop in properties.items():
            if prop.get(PROP_ENUM): 
                values = self.get_permissive_values(prop.get(PROP_ENUM))
            
            else: 
                if (prop.get(PROP_TYPE) and isinstance(prop.get(PROP_TYPE), dict) and prop.get(PROP_TYPE).get(VALUE_TYPE) in prop_list_types) and prop.get(PROP_TYPE).get(PROP_ENUM):
                    values = self.get_permissive_values(prop.get(PROP_TYPE).get(PROP_ENUM))
                else: continue

            if values and len(values) > 0:
                permissive_value_dic[prop_name] = values
                
            else: 
                msg = f'No permissive value for the property: {prop_name} in {model_props_file}!'
  
        return True, permissive_value_dic, msg

    def get_permissive_values(self, enum_value):
        permissive_values = None
        if enum_value and len(enum_value) > 0:
            permissive_values = _get_item_type(enum_value)
        return permissive_values
    
    """
    get properties of a model by datacommon and version
    """
    def get_model_props(self, model_key):
        pass

def model_key(data_common, version):
    return f"{data_common.upper()}_{version}"

def _get_item_type(prop_enum):
    enum = set()
    for t in prop_enum:
        if not re.search(r'://', t):
            enum.add(t)
    if len(enum) > 0:
        return list(enum)
    else:
        return None


   
    
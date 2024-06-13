import tarfile, json
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

def get_data_time(format = "%Y-%m-%d-%H-%M-%S-%f"):
    """
    get current time in format
    """  
    return datetime.strftime(datetime.now(), format)[:-3]


import os
from ruamel.yaml import YAML
yaml = YAML()

def get_yml_files(path):
    """
    Returns a list of all the yml files in the path.

    Args:
        path (str): path to directory with files

    Returns:
        list: list of paths to .yml files
    """
    out_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".yml"):
                out_files.append(os.path.join(root, file))
    
    return out_files

def extract_None_vals(var, keys2check, path=None):
    """
    Recursive function that identifies the values to a given key within a nested dict,
    and returns the nested keys along with the value and an error message when encountering
    a None value for a key in keys2check.

    Args:
        var (dict): The dict to search in.
        keys2check (set): set of key names to be checked
        path (list): The current path of keys.

    Yields:
        gen: Generator object with tuples of the full path to each occurrence of the key and its value.
    """

    if path is None:
        path = []

    if hasattr(var, 'items'):
        for k, v in var.items():
            new_path = path + [k]

            if k in keys2check and v == None:
                yield(new_path, v, 'cannot be None.')
                            
            if isinstance(v, dict):
                for result in extract_None_vals(v, keys2check, new_path):
                    yield result

def test_None():
    """
    pytest function to check that the values to the keys in keys2check are not None.
    """
    keys2check = ('qrx','qmr','wb_specnet','specnet','qradio','autoconf','qdra-inst1','qdra-inst2','proxy')
    paths = ['./v1', './v2']

    yml_files = []
    for path in paths:
        yml_files = [*yml_files, *get_yml_files(path)]

    print(f'Running tests for .yml files in {paths}:')

    isError = False
    badConfigs = []

    for yml_file in yml_files:
        with open(yml_file, 'r') as stream:

            f = yaml.load(stream)
            
            failed_keys = extract_None_vals(f, keys2check)
            for i, (path, value, msg) in enumerate(failed_keys):
                    if i==0:
                        isError = True
                        badConfigs.append(yml_file)

                    print(f"{yml_file}: {' -> '.join(path)}: {value}, {msg}")

        if yml_file in badConfigs:
            print()
            
    assert not isError, f'{len(set(badConfigs))} bad configs: {set(badConfigs)}'
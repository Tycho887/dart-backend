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

def extract_bad_tags(key, var, path=None):
    """
    Recursive function that identifies the values to a given key within a nested dict,
    and returns the nested keys along with the value if it is NOT a list.

    Args:
        key (str): The key to look for.
        var (dict): The dict to search in.
        path (list): The current path of keys.

    Yields:
        gen: Generator object with tuples of the full path to each occurrence of the key and its value.
    """
    if path is None:
        path = []

    if hasattr(var, 'items'):
        for k, v in var.items():
            new_path = path + [k]
            if k == key:
                if not isinstance(v, list):
                    yield (new_path, v)
            if isinstance(v, dict):
                for result in extract_bad_tags(key, v, new_path):
                    yield result

def test_tags():
    """
    pytest function to check that all tags fields in a config are read as list
    """
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

            failed_tags = extract_bad_tags('tags', f)

            for i, (path, value) in enumerate(failed_tags):
                if i==0:
                    isError = True
                    badConfigs.append(yml_file)
                print(f"{yml_file}: {' -> '.join(path)}: {value}, is not a list.")

        if yml_file in badConfigs:
            print()

    assert not isError, f'{len(set(badConfigs))} bad configs: {set(badConfigs)}'
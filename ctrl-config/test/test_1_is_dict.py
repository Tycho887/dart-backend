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

def test_is_dict():
    """
    pytest function to check that all configs can be read as dict
    using ruamel.yaml to parse the file includes a duplicate check
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

            # check if file can be read as dict without duplicates
            try:
                f = yaml.load(stream)

            except Exception as e:
                isError = True
                badConfigs.append(yml_file)
                print(f'{yml_file}: Parsing failed with msg: {e}')

            else:
                if not isinstance(f, dict):
                    isError = True
                    badConfigs.append(yml_file)
                    print(f'{yml_file}: is not a dict')

        if yml_file in badConfigs:
            print()
            
    assert not isError, f'{len(set(badConfigs))} bad configs: {set(badConfigs)}'
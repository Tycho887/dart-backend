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

def extract_bad_mtx_tags(key, var, way, path=None):
    """
    Recursive function that identifies the values to a given key within a nested dict,
    and returns the nested keys along with the value and an error message.

    Args:
        key (str): The key to look for.
        var (dict): The dict to search in.
        way (str): 'ul' or 'dl' for uplink or downlink respectively.
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
                    yield (new_path, v, 'is not a list.')
                else:
                    if not v == []:
                        if way == 'ul': # for uplink routes
                            if isinstance(v[0], list): # check whether v is a nested list
                                if not (len(v) == 1 and len(v[0]) == 2): # v can only contain 1 list of length 2
                                    yield (new_path, v, 'must be be an empty list, a list of length 2, or a nested list containing 1 list of length 2')
                            elif not len(v) == 2: # must be list of length 2 if not nested list
                                yield (new_path, v, 'must be be an empty list, a list of length 2, or a nested list containing 1 list of length 2')

                        elif way == 'dl': # for downlink routes
                            route_lens = [len(route)==2 for route in v]
                            if not all(route_lens): # check that all given downlink routes are a list of length 2
                                yield (new_path, v, 'must be be an empty list or a list of lists with length 2 each')
                        else:
                            raise Exception('way not specified must be ul or dl')
                            
            if isinstance(v, dict):
                for result in extract_bad_mtx_tags(key, v, way, new_path):
                    yield result

def test_mtx_tags_uplink_downlink():
    """
    pytest function for mission configs that checks that mtx_tags_downlink and mtx_tags_downlink are correct Type,
    i.e., empty list or list with length 2 OR empty list or list of list with length 2 each, respectively.
    """
    paths = ['./v1/mission/', './v2/mission/']

    yml_files = []
    for path in paths:
        yml_files = [*yml_files, *get_yml_files(path)]

    print(f'Running tests for .yml files in {paths}:')

    isError = False
    badConfigs = []

    for yml_file in yml_files:
        with open(yml_file, 'r') as stream:

            f = yaml.load(stream)

            # check that all mtx_tags_downlink fields are read as list
            failed_tags = extract_bad_mtx_tags('mtx_tags_downlink', f, 'dl')
            for i, (path, value, msg) in enumerate(failed_tags):
                    if i==0:
                        isError = True
                        badConfigs.append(yml_file)

                    print(f"{yml_file}: {' -> '.join(path)}: {value}, {msg}")

            failed_tags = extract_bad_mtx_tags('mtx_tags_uplink', f, 'ul')
            for i, (path, value, msg) in enumerate(failed_tags):
                    if i==0:
                        isError = True
                        badConfigs.append(yml_file)

                    print(f"{yml_file}: {' -> '.join(path)}: {value}, {msg}")

        if yml_file in badConfigs:
            print()
    
    assert not isError, f'{len(set(badConfigs))} bad mission configs: {set(badConfigs)}'
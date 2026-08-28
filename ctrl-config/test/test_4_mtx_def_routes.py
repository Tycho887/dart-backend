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

def test_mtx_def_routes():
    """
    pytest function for system configs that currently checks:
    1. uplink_mtx def_routes is correct Type, i.e., empty list, list with length 2, or a nested list containing 1 list of length 2.
    2. downlink_mtx def_routes is correct Type, i.e., empty list or list of list with length 2 each
    """

    paths = ['./v1/system/', './v2/system/']

    yml_files = []
    for path in paths:
        yml_files = [*yml_files, *get_yml_files(path)]

    print(f'Running tests for .yml files in {paths}:')

    isError = False
    badConfigs = []

    for yml_file in yml_files:
        with open(yml_file, 'r') as stream:

            f = yaml.load(stream)

            # check that def_routes fields are read correctly
            # check uplink_mtx
            try:
                ul_routes = f['defaults']['uplink_mtx']['def_routes']
            except:
                pass # if defaults -> uplink_mtx -> def_routes not present in dict
            else:
                # raise error if def_routes is not a list
                if not isinstance(ul_routes, list):
                    isError = True
                    badConfigs.append(yml_file)
                    print(f'{yml_file}: defaults -> uplink_mtx -> def_routes: {ul_routes}, is not a list.')
                else:
                    # raise error if def_routs is not an empty list or a list of length 2
                    if not ul_routes == []:
                        if isinstance(ul_routes[0], list): # check whether ul_routes is nested list
                            if not (len(ul_routes) == 1 and len(ul_routes[0]) == 2): # check that only 1 nested list with length 2
                                isError = True
                                badConfigs.append(yml_file)
                                print(f'{yml_file}: defaults -> uplink_mtx -> def_routes: {ul_routes}, must be an empty list, a list of length 2, or a nested list containing 1 list of length 2.')

                        elif not len(ul_routes) == 2: # check that length is 2
                            isError = True
                            badConfigs.append(yml_file)
                            print(f'{yml_file}: defaults -> uplink_mtx -> def_routes: {ul_routes}, must be an empty list, a list of length 2, or a nested list containing 1 list of length 2.')

            # check downlink_mtx
            try:
                dl_routes = f['defaults']['downlink_mtx']['def_routes']
            except:
                pass # if defaults -> downlink_mtx -> def_routes not present in dict
            else:
                # raise error if def_routes is not a list
                if not isinstance(dl_routes, list):
                    isError = True
                    badConfigs.append(yml_file)
                    print(f'{yml_file}: defaults -> downlink_mtx -> def_routes: {dl_routes}, is not a list.')
                else:
                    # check whether def_routes is an empty list 
                    if not dl_routes == []:
                        # for non-empty list raise error if def_routes is not a list of lists of length 2
                        route_lens = [len(route) == 2 for route in dl_routes]
                        if not all(route_lens):
                            isError = True
                            badConfigs.append(yml_file)
                            print(f'{yml_file}: defaults -> downlink_mtx -> def_routes: {dl_routes}, must be an empty list or a list of lists with length 2 each.')
            
        if yml_file in badConfigs:
            print()

    assert not isError, f'{len(set(badConfigs))} bad system configs: {set(badConfigs)}'
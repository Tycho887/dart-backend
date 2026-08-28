import os
import yaml

yml_files = []

for root, dirs, files in os.walk("./v1"):
    for file in files:
        if file.endswith(".yml"):
            yml_files.append(os.path.join(root, file))
            
for root, dirs, files in os.walk("./v2"):
    for file in files:
        if file.endswith(".yml"):
            yml_files.append(os.path.join(root, file))

            
for yml_file in yml_files:
    with open(yml_file, 'r') as stream:
        f = yaml.safe_load(stream)
        print(f'Loading {yml_file} and checking if it is a dict..')
        assert isinstance(f, dict)        

        

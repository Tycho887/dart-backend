#!/usr/bin/env python3
from pathlib import Path
import yaml

def deploy_info_to_icmp_heartbeat(data):

    system = data.get('system')
    host = data.get('host')

    d = {
        'type': 'icmp',
        'name': f'Controller {system}',
        'id': f'controller-{system}',
        'hosts': [host],
        'tags': 'controller',
        'scheduler': '*/5 * * * * * *',
    }

    return d
    


def open_yaml(path):

    with open(path, 'r') as handler:
        return yaml.safe_load(handler)

config_dir = Path('../../')

search_paths = ['system', 'analyzer']
icmps = []

for path in search_paths:
    p = config_dir / 'v3' / path
    #print(p)

    files = p.glob('*.yml')

    for file in files:

        data = open_yaml(file) or {}

        deploy_info = data.get('deploy_info')
        if deploy_info:
            d = deploy_info_to_icmp_heartbeat(deploy_info)
            icmps.append(d)

print(icmps)


with open('autogen_deployment_icmp_heartbeats.yml', 'w') as handler:
    yaml.dump(icmps, handler)
            
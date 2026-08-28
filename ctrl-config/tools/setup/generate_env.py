#!/usr/bin/env python3
"""
Script to generate an system.env file
"""
from pathlib import Path
import os
import sys
import yaml
from urllib.parse import urlparse


GLOBAL_DEFAULTS = {
    'CONTROLLER_IMAGE': 'registry.gitlab.ksat.no/lite/antenna_stack/controller/stable:v1.5.0',
#    'CONTROLLER_IMAGE': 'registry.gitlab.ksat.no/lite/antenna_stack/controller/stable:v1.5.0',
#    'CONTROLLER_IMAGE': 'registry.gitlab.ksat.no/lite/antenna_stack/controller/stable:v1.5.0',
#    'CONTROLLER_IMAGE': 'registry.gitlab.ksat.no/lite/antenna_stack/controller/stable:v1.5.0',    
}

yml_files = []
for root, dirs, files in os.walk("../../v2/system"):
    for file in files:
        if file.endswith(".yml"):
            yml_files.append(os.path.join(root, file))

       
systems = {}

for yml_file in yml_files:
    try:
        with open(yml_file, 'r') as stream:
        
            f = yaml.safe_load(stream)
        
            p = Path(yml_file)
                
            systems[p.stem] = f
    except:
        pass

local = {}

try:
    with open('local.yml', 'r') as stream:
        f = yaml.safe_load(stream)        
        if isinstance(f, dict):
            local = f

except Exception as e:
    print(e)

    
_system = sys.argv[1]
system = systems[_system]

defaults = system.get('defaults')

HOSTS = ['qradio', 'specnet', 'qmr']

with open('global.yml', 'r') as stream:
    DEFAULTS = yaml.safe_load(stream)

print(DEFAULTS)

_env = []

def_images = DEFAULTS.get('IMAGES')

for sid, default in defaults.items():
    if sid in HOSTS:
        rest = urlparse(default.get('rest'))
        s = f'export {sid.upper()}_HOST={rest.hostname}'
        _env.append(s)

    if sid == 'proxy':
        proxy_host = default.get('access')
        s = f'export PROXY_HOST={proxy_host}'
        _env.append(s)

        proxy_mode = default.get('mode')
        s = f'export PROXY_MODE={proxy_mode}'
        _env.append(s)

        proxy_host = default.get('access')
        s = f'export PROXY_HOST={proxy_host}'
        _env.append(s)

        value = default.get('access')
        s = f'export MC_HOST={value}'
        _env.append(s)

        value = default.get('access')
        s = f'export SPECTRUM_HOST={value}'
        _env.append(s)
                
        tgts = ['ACCESS_PROXY', 'CONTROL_API', 'SPECTRUM_PLOT']

        images = default.get('images') or {}
        for tgt in tgts:
            image = images.get(tgt) or def_images.get(tgt)
            
            if not image:
                continue
            
            env_value = f'{tgt}_IMAGE={image}'
            _env.append(f'export {env_value}')
                        
    if sid == 'controller':
        session = default.get('session')
        s = f'export SESSION={session}'
        _env.append(s)

        auto_pull = default.get('auto_pull')
        s = f'export AUTO_PULL={auto_pull}'
        _env.append(s)

        images = default.get('images') or {}
        key = 'CONTROLLER'
        controller_image = images.get(key) or DEFAULTS.get('IMAGES').get(key)
        s = f'CONTROLLER_IMAGE={controller_image}'
        _env.append(f'export {s}')

        key = 'SCHEDULER'
        scheduler_image = images.get(key) or DEFAULTS.get('IMAGES').get(key)
        s = f'SCHEDULER_IMAGE={scheduler_image}'
        _env.append(f'export {s}')

        dispatchers = default.get('dispatchers')

        for did, dispatcher in dispatchers.items():
            s = f'DISPATCHERS={did}={dispatcher}'
            _env.append(f'export {s}')
            
        site_id = default.get('site_id') or 36136461
        s = f'SITE_ID={site_id}'
        _env.append(f'export {s}')

        antenna_id = default.get('antenna_id')
        s = f'ANTENNA_IDS={antenna_id}'
        _env.append(f'export {s}')

        mode = default.get('mode') or 'PRODUCTION'

        s = f'MODE={mode}'

        _env.append(f'export {s}')

        schedule_ahead = default.get('schedule_ahead') or 10
        s = f'SCHEDULE_AHEAD={schedule_ahead}'

        _env.append(f'export {s}')

        query_frequency = default.get('query_frequency') or 2
        s = f'QUERY_FREQUENCY={query_frequency}'

        _env.append(f'export {s}')
        

for key, value in local.items():
    s = f'export {key}={value}'
    _env.append(s)
        
st = '\n'.join(_env)

out = Path(f'{_system}.env')
out.write_text(st)

#!/usr/bin/env python3
from pathlib import Path
import os
import yaml
import jinja2

TEMPLATES = 'templates'
BUILD = Path('../source/proxy.rst')

yml_files = []
            
for root, dirs, files in os.walk("../../v2/system"):
    for file in files:
        if file.endswith(".yml"):
            f = Path(os.path.join(root, file))
            yml_files.append(f)

systems = {}

for yml_file in yml_files:

    try:
        with open(yml_file, 'r') as stream:
            f = yaml.safe_load(stream)
            p = Path(yml_file)
            systems[p.stem] = f

    except:
        pass
        
results = []
for system, cfg in systems.items():
    
    defaults = cfg.get('defaults')
    proxy = defaults.get('proxy')
    if not proxy:
        continue
        
    site = proxy.get('site')
    site_sdwan = proxy.get('site_sdwan')
    access = proxy.get('access')
    mode = proxy.get('mode')
    operational = proxy.get('operational') or False
    
    if not site and not access:
        continue

    results.append((system, site, site_sdwan, access, mode, operational))
    
    print(system, site, site_sdwan, access, mode, proxy)


results.sort(key=lambda tup: tup[0])  # sorts in place
    
template_loader = jinja2.FileSystemLoader(searchpath=TEMPLATES)
template_env = jinja2.Environment(loader=template_loader)
template = template_env.get_template("proxy.jinja2")

output = template.render(results=results)

print(output)

BUILD.write_text(output)


    

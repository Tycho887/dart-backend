#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import glob
from schemas import build_spacecraft_link_validator, load_yaml


files: list = glob.glob('v2/spacecrafts/*.yml')
file_paths: list = [Path(file) for file in files]
spacecraft_names: list = [file_path.stem for file_path in file_paths]
spacecraft_names.sort()

link_validator = build_spacecraft_link_validator()

validation_errors: int = 0
validation_warnings: int = 0
spacecrafts_without_links: int = 0


for spacecraft_name in spacecraft_names:
    print(f'Checking {spacecraft_name}..')

    if spacecraft_name.lower() in ['template', 'example']:
        print(f'..{spacecraft_name} OK!')
        continue

    spacecraft = load_yaml(Path(f'v2/spacecrafts/{spacecraft_name}.yml'))

    if not isinstance(spacecraft.get('links'), dict):
        print(f'..{spacecraft_name} NOT OK!')
        print('Links not a dict')
        validation_errors += 1
        continue

    if not len(spacecraft.get('links')) > 0:
        print(f'{spacecraft_name} has no links!')
        spacecrafts_without_links += 1
        
    for link_name, link in spacecraft.get('links', {}).items():

        link_name_parts = link_name.split('_')

        if not len(link_name_parts) == 5:
            print(f'..{spacecraft_name} NOT OK!')
            print(f'Link name {link_name} not valid, link name should be on form <band>_band>_<direction>_p<physical_quantity>_<counter>. Example: x_band_downlink_p1_1')

            if link_name_parts[0] not in ['x', 's', 'k']:
                print(f'Band {link_name_parts[0]} not valid, band should be one of x, s or k')

            if link_name_parts[1] not in ['band']:
                print(f'Band {link_name_parts[1]} not valid, band should be band')

            if link_name_parts[2] not in ['uplink', 'downlink']:
                print(f'Direction {link_name_parts[2]} not valid, direction should be one of uplink or downlink')

            if link_name_parts[3] not in ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']:
                print(f'Physical quantity {link_name_parts[3]} not valid, physical quantity should be one of p1, p2, p3, p4, p5, p6 or p7')

            if not link_name_parts[4].isdigit():
                print(f'Counter {link_name_parts[4]} not valid, counter should be a number')

            validation_errors += 1
            continue

        try:
            link_validator.validate(link)

            if not link.get('polarization'):
                print(f'..{spacecraft_name}:{link_name} missing polarization!')
                validation_warnings += 1

        except Exception as e:
            validation_errors += 1
            print(f'..{spacecraft_name} NOT OK!')
            print(e)
            print(link)
            continue

    if validation_errors:
        break

if validation_errors > 0:
    print(f'Found {validation_errors} validation errors!')
    exit(1)

if validation_warnings > 0:
    print(f'Found {validation_warnings} validation warnings!')
    exit(1)

if spacecrafts_without_links > 0:
    print(f'Found {spacecrafts_without_links} spacecrafts without links!')
    exit(1)

else:
    print('All OK!')
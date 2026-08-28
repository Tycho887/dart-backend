#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from time import sleep
from pathlib import Path
import glob
import json
from requests import Session


def parse_antenna_data(text: str) -> dict:
    """_summary_

    Args:
        text (str): _description_

    Returns:
        dict: _description_
    """
    splitted = text.strip().splitlines()
    comments_removed = []
    profile_position = None

    for line in splitted:
        blank_line = line.strip().replace(' ', '')

        if blank_line == '':
            continue
        if not blank_line.startswith('#'):
            comments_removed.append(blank_line)

    for idx, line in enumerate(comments_removed):
        if line == 'Azimuth\tHorizon':
            profile_position = idx
            break

    comments_removed

    d = dict(s.split('=', 1) for s in comments_removed[0:6])

    knosweb_antenna_file: dict = {
        'antenna': d.get('Antenna'),
        'latitude': float(d.get('Latitude')),
        'longitude': float(d.get('Longitude')),
        'altitude': float(d.get('Altitude')),
        'horizon': float(d.get('Horizon')),
        'profile': True if d.get('Profile') == 'Yes' else False,
        'horizon_profile': {}
    }

    if knosweb_antenna_file['profile'] == True:
        for line in comments_removed[profile_position+1:]:
            _splitted = line.split('\t')
            knosweb_antenna_file['horizon_profile'][float(_splitted[0])] = float(_splitted[1])

    else:
        knosweb_antenna_file['horizon_profile'] = None

    return knosweb_antenna_file


# Set up the session
s = Session()

base_url: str = 'http://knosweb.tss.no/sgp4/antenna.'

files: list = glob.glob('v2/system/*.yml')
file_paths: list = [Path(file) for file in files]
antenna_names: list = [file_path.stem for file_path in file_paths]

antenna_data: dict = {}

{
    'status_code': None,
    'text': None
}

for antenna_name in antenna_names:
    print(f'Checking {antenna_name}..')
    sleep(0.1)
    antenna_data[antenna_name] = {}

    url: str = base_url + antenna_name
    response = s.get(url, timeout=10)
    print(response.status_code)

    antenna_data[antenna_name]['status_code'] = response.status_code
    if response.ok:
        print(f'..{antenna_name} OK!')
        antenna_data[antenna_name]['text'] = response.text
    else:
        print(f'..{antenna_name} NOT OK!')
        antenna_data[antenna_name]['text'] = None

output_dir = Path('v2/antenna_horizons')
output_dir.mkdir(parents=True, exist_ok=True)

for antenna_name, d in antenna_data.items():

    if not d.get('text'):
        continue

    print(f'Parsing {antenna_name}..')

    try:
        parsed = parse_antenna_data(d.get('text'))
    except Exception as e:
        print(f'..{antenna_name} NOT PARSED!')
        print(e)
        print(antenna_data)
        continue

    with open(output_dir / f'{antenna_name}.json', 'w') as f:
        json.dump(parsed, f)

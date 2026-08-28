#!/usr/bin/env BASH

echo $SESSION
python3 generate_env.py $1
source $1.env
cd bundle
docker-compose up -d

#!/usr/bin/env BASH

python3 generate_env.py $1
source $1.env
cd proxy
docker-compose up -d

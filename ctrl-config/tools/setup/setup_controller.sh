#!/usr/bin/env BASH

echo $SESSION
python3 generate_env.py $1
source $1.env
cd controller
docker-compose up -d

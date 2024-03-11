#!/usr/bin/env bash

python3 -m venv --upgrade-deps ./env > /dev/null && echo "env created"
source ./env/bin/activate && echo "env activated"
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git

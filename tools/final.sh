#!/usr/bin/env bash

python tools/process_data/generate_testB_json.py

PORT=${PORT:-29400}

python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    tools/test.py configs/final.py final-353579b9.pth --launcher pytorch --format-only --eval-options "jsonfile_prefix=final"

python tools/process_data/json2submit.py

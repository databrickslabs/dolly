#!/bin/bash

deepspeed \
     --include="localhost:2,3" \
     --module training.trainer \
     --deepspeed config/ds_z3_bf16_config.json \
     --epochs 10 \
     --local-output-dir /home/bo_ling/dolly_training/ma_test \
     --local-data-file-path /home/bo_ling/dolly/ma_data/so3_long.jsonl \
     --per-device-train-batch-size 2 \
     --per-device-eval-batch-size 2 \
     --test-size 50 \
     --lr 1e-5

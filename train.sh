#!/bin/bash

deepspeed \
     --include="localhost:3,2" \
     --module training.trainer \
     --deepspeed config/ds_z3_bf16_config.json \
     --epochs 1 \
     --local-output-dir /home/bo_ling/dolly/dolly_training \
     --per-device-train-batch-size 2 \
     --per-device-eval-batch-size 2 \
     --lr 1e-5

#!/bin/bash

deepspeed --num_gpus=2 \
     --module training.trainer \
     --deepspeed config/ds_z3_bf16_config.json \
     --epochs 1 \
     --local-output-dir dolly_training \
     --per-device-train-batch-size 8 \
     --per-device-eval-batch-size 8 \
     --lr 1e-5

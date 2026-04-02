#!/usr/bin/env bash
set -e
cd /workspace/TRELLIS
python train.py \
  --config /workspace/TRELLIS/configs/arch1_stage1_smoke.json \
  --output_dir /workspace/arch1_runs/stage1_smoke \
  --data_dir /workspace/arch1_preprocessed \
  --num_gpus 1

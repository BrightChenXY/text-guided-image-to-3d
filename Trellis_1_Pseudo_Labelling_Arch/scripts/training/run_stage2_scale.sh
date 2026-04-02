#!/usr/bin/env bash
set -e
cd /workspace/TRELLIS
CONFIG=${1:-/workspace/TRELLIS/configs/arch1_stage2_poc.json}
DATA_DIR=${2:-/workspace/arch1_preprocessed}
OUT_DIR=${3:-/workspace/arch1_runs/stage2_poc}
python train.py \
  --config "$CONFIG" \
  --output_dir "$OUT_DIR" \
  --data_dir "$DATA_DIR" \
  --num_gpus 1

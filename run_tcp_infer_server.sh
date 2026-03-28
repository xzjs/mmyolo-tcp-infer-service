#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/projects/piping/src:${PYTHONPATH}"

HOST="${TCP_HOST:-0.0.0.0}"
PORT="${TCP_PORT:-9000}"

DEFAULT_CONFIG="${MODEL_CONFIG:-./projects/piping/configs/yolov8_n_fast_8xb16-500e_defect_lower_lr_ag.py}"
DEFAULT_CHECKPOINT="${MODEL_CHECKPOINT:-./weights/best_coco_破裂_precision_epoch_25.pth}"
DEFAULT_DEVICE="${MODEL_DEVICE:-cuda:0}"
DEFAULT_SCORE_THR="${SCORE_THR:-0.3}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/tcp_results}"
INFER_STRIDE="${INFER_STRIDE:-3}"
MAX_FRAMES="${MAX_FRAMES:--1}"
PROGRESS_EVERY_FRAMES="${PROGRESS_EVERY_FRAMES:-30}"

python tools/piping_infer/tcp_infer_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --default-config "$DEFAULT_CONFIG" \
  --default-checkpoint "$DEFAULT_CHECKPOINT" \
  --default-device "$DEFAULT_DEVICE" \
  --default-score-thr "$DEFAULT_SCORE_THR" \
  --default-out-dir "$OUTPUT_DIR" \
  --default-infer-stride "$INFER_STRIDE" \
  --default-max-frames "$MAX_FRAMES" \
  --default-progress-every "$PROGRESS_EVERY_FRAMES"

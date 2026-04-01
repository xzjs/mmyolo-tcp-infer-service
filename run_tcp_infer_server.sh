#!/usr/bin/env bash

set -e

# 避免部分环境里 OMP_NUM_THREADS 非法导致 libgomp 报错
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "$(dirname "$0")"

export PYTHONPATH="$PWD:$PWD/projects/piping/src:${PYTHONPATH}"

HOST="${TCP_HOST:-0.0.0.0}"
PORT="${TCP_PORT:-9000}"
DEFAULT_CONFIG="${MODEL_CONFIG:-./projects/piping/configs/yolov8_s_fast_8xb16-500e_ours.py}"
DEFAULT_CHECKPOINT="${MODEL_CHECKPOINT:-./weights/best_coco_破裂_precision_epoch_25.pth}"
DEFAULT_DEVICE="${MODEL_DEVICE:-cuda:0}"
DEFAULT_SCORE_THR="${SCORE_THR:-0.3}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/tcp_results}"

INFER_STRIDE="${INFER_STRIDE:-3}"
VIS_STRIDE="${VIS_STRIDE:-3}"
WRITE_STRIDE="${WRITE_STRIDE:-3}"
RESIZE_WIDTH="${RESIZE_WIDTH:-1280}"
RESIZE_HEIGHT="${RESIZE_HEIGHT:-720}"
VIS_SCORE_THR="${VIS_SCORE_THR:-0.18}"
VIS_TOPK="${VIS_TOPK:-10}"
MATCH_IOU_THR="${MATCH_IOU_THR:-0.35}"
MATCH_CENTER_DIST_RATIO="${MATCH_CENTER_DIST_RATIO:-0.22}"
MAX_FRAME_GAP="${MAX_FRAME_GAP:-9}"
POST_MERGE_GAP_FRAMES="${POST_MERGE_GAP_FRAMES:-9}"
POST_MERGE_IOU_THR="${POST_MERGE_IOU_THR:-0.30}"
CHAIN_GAP_FRAMES="${CHAIN_GAP_FRAMES:-18}"
CHAIN_IOU_THR="${CHAIN_IOU_THR:-0.18}"
CHAIN_CENTER_DIST_RATIO="${CHAIN_CENTER_DIST_RATIO:-0.28}"
CHAIN_MAX_KEEP="${CHAIN_MAX_KEEP:-1}"
MIN_HITS="${MIN_HITS:-2}"
SINGLE_HIT_HIGH_SCORE="${SINGLE_HIT_HIGH_SCORE:-0.45}"
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
  --default-vis-stride "$VIS_STRIDE" \
  --default-write-stride "$WRITE_STRIDE" \
  --default-resize-width "$RESIZE_WIDTH" \
  --default-resize-height "$RESIZE_HEIGHT" \
  --default-vis-score-thr "$VIS_SCORE_THR" \
  --default-vis-topk "$VIS_TOPK" \
  --default-match-iou-thr "$MATCH_IOU_THR" \
  --default-match-center-dist-ratio "$MATCH_CENTER_DIST_RATIO" \
  --default-max-frame-gap "$MAX_FRAME_GAP" \
  --default-post-merge-gap-frames "$POST_MERGE_GAP_FRAMES" \
  --default-post-merge-iou-thr "$POST_MERGE_IOU_THR" \
  --default-chain-gap-frames "$CHAIN_GAP_FRAMES" \
  --default-chain-iou-thr "$CHAIN_IOU_THR" \
  --default-chain-center-dist-ratio "$CHAIN_CENTER_DIST_RATIO" \
  --default-chain-max-keep "$CHAIN_MAX_KEEP" \
  --default-min-hits "$MIN_HITS" \
  --default-single-hit-high-score "$SINGLE_HIT_HIGH_SCORE" \
  --default-max-frames "$MAX_FRAMES" \
  --default-progress-every "$PROGRESS_EVERY_FRAMES"

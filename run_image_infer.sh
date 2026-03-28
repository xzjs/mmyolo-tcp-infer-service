#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/projects/piping/src:${PYTHONPATH}"

IMG_PATH="${IMG_PATH:-./demo/demo.jpg}"
CONFIG_PATH="${CONFIG_PATH:-./projects/piping/configs/yolov8_n_fast_8xb16-500e_defect_lower_lr_ag.py}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./weights/best_coco_破裂_precision_epoch_25.pth}"
DEVICE="${DEVICE:-cuda:0}"
SCORE_THR="${SCORE_THR:-0.3}"
OUT_DIR="${OUT_DIR:-./output/images}"
OUT_JSONL="${OUT_JSONL:-./output/images/predictions.jsonl}"

python tools/piping_infer/image_infer_piping.py \
  "$IMG_PATH" \
  "$CONFIG_PATH" \
  "$CHECKPOINT_PATH" \
  --device "$DEVICE" \
  --score-thr "$SCORE_THR" \
  --out-dir "$OUT_DIR" \
  --out-jsonl "$OUT_JSONL"

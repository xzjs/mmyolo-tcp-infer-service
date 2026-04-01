#!/usr/bin/env bash

set -e

# 避免部分环境里 OMP_NUM_THREADS 非法导致 libgomp 报错
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# 进入仓库根目录
cd "$(dirname "$0")"

# 让自定义模块可以被配置文件里的 custom_imports 找到
export PYTHONPATH="$PWD:$PWD/projects/piping/src:${PYTHONPATH}"

# ------------------------------
# 基础模型参数
# ------------------------------
VIDEO_PATH="${VIDEO_PATH:-./input_videos/2JFLHLJ667_2JFLHLJ670_20230315.mp4}"
CONFIG_PATH="${CONFIG_PATH:-./projects/piping/configs/yolov8_s_fast_8xb16-500e_ours.py}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./weights/best_coco_破裂_precision_epoch_25.pth}"
DEVICE="${DEVICE:-cuda:0}"
SCORE_THR="${SCORE_THR:-0.3}"

# ------------------------------
# 输出路径
# ------------------------------
OUT_VIDEO="${OUT_VIDEO:-./output/video_result.mp4}"
OUT_JSONL="${OUT_JSONL:-./output/video_result.jsonl}"
OUT_DEFECTS_JSON="${OUT_DEFECTS_JSON:-./output/video_result_defects.json}"
OUT_FRAME_INDEX_JSON="${OUT_FRAME_INDEX_JSON:-./output/video_result_frame_index.json}"
DEFECT_FRAMES_DIR="${DEFECT_FRAMES_DIR:-./output/video_result_defect_images}"
# 兼容旧参数：当前版本不再导出 crop 图。
DEFECT_CROPS_DIR="${DEFECT_CROPS_DIR:-}"

# ------------------------------
# 抽帧 / 可视化参数
# ------------------------------
INFER_STRIDE="${INFER_STRIDE:-3}"
VIS_STRIDE="${VIS_STRIDE:-3}"
WRITE_STRIDE="${WRITE_STRIDE:-3}"
RESIZE_WIDTH="${RESIZE_WIDTH:-1280}"
RESIZE_HEIGHT="${RESIZE_HEIGHT:-720}"
VIS_SCORE_THR="${VIS_SCORE_THR:-0.18}"
VIS_TOPK="${VIS_TOPK:-10}"
PRINT_EVERY="${PRINT_EVERY:-30}"
MAX_FRAMES="${MAX_FRAMES:--1}"

# ------------------------------
# 去重参数
# ------------------------------
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

python tools/piping_infer/video_infer_piping.py \
  "$VIDEO_PATH" \
  "$CONFIG_PATH" \
  "$CHECKPOINT_PATH" \
  --device "$DEVICE" \
  --score-thr "$SCORE_THR" \
  --out-video "$OUT_VIDEO" \
  --out-jsonl "$OUT_JSONL" \
  --out-defects-json "$OUT_DEFECTS_JSON" \
  --out-frame-index-json "$OUT_FRAME_INDEX_JSON" \
  --save-defect-frames-dir "$DEFECT_FRAMES_DIR" \
  --save-defect-crops-dir "$DEFECT_CROPS_DIR" \
  --infer-stride "$INFER_STRIDE" \
  --vis-stride "$VIS_STRIDE" \
  --write-stride "$WRITE_STRIDE" \
  --resize-width "$RESIZE_WIDTH" \
  --resize-height "$RESIZE_HEIGHT" \
  --vis-score-thr "$VIS_SCORE_THR" \
  --vis-topk "$VIS_TOPK" \
  --match-iou-thr "$MATCH_IOU_THR" \
  --match-center-dist-ratio "$MATCH_CENTER_DIST_RATIO" \
  --max-frame-gap "$MAX_FRAME_GAP" \
  --post-merge-gap-frames "$POST_MERGE_GAP_FRAMES" \
  --post-merge-iou-thr "$POST_MERGE_IOU_THR" \
  --chain-gap-frames "$CHAIN_GAP_FRAMES" \
  --chain-iou-thr "$CHAIN_IOU_THR" \
  --chain-center-dist-ratio "$CHAIN_CENTER_DIST_RATIO" \
  --chain-max-keep "$CHAIN_MAX_KEEP" \
  --min-hits "$MIN_HITS" \
  --single-hit-high-score "$SINGLE_HIT_HIGH_SCORE" \
  --print-every "$PRINT_EVERY" \
  --max-frames "$MAX_FRAMES"

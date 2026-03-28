#!/usr/bin/env bash
set -e

# 进入仓库根目录。
cd "$(dirname "$0")"

# 让自定义模块可以被配置文件里的 custom_imports 找到。
export PYTHONPATH="$PWD:$PWD/projects/piping/src:${PYTHONPATH}"

# 这里给出默认值；你也可以在命令行用环境变量覆盖。
# 模型路径和名称
VIDEO_PATH="${VIDEO_PATH:-./input_videos/2JFLHLJ667_2JFLHLJ670_20230315.mp4}"
CONFIG_PATH="${CONFIG_PATH:-./projects/piping/configs/yolov8_s_fast_8xb16-500e_ours.py}"

# CONFIG_PATH="${CONFIG_PATH:-./projects/piping/configs/yolov8_n_fast_8xb16-500e_defect_lower_lr_ag.py}"



# 模型权重
# CHECKPOINT_PATH="${CHECKPOINT_PATH:-./weights/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./weights/best_coco_破裂_precision_epoch_25.pth}"


DEVICE="${DEVICE:-cuda:0}"
SCORE_THR="${SCORE_THR:-0.3}"
OUT_VIDEO="${OUT_VIDEO:-./output/video_result322.mp4}"
OUT_JSONL="${OUT_JSONL:-./output/video_result322.jsonl}"

python tools/piping_infer/video_infer_piping.py \
  "$VIDEO_PATH" \
  "$CONFIG_PATH" \
  "$CHECKPOINT_PATH" \
  --device "$DEVICE" \
  --score-thr "$SCORE_THR" \
  --out-video "$OUT_VIDEO" \
  --out-jsonl "$OUT_JSONL"

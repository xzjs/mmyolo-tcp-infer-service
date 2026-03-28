#!/usr/bin/env bash
set -e

# 进入仓库根目录。
cd "$(dirname "$0")"

# 这个安装脚本默认你已经在 AutoDL 的 GPU 环境中，且已经有合适版本的 PyTorch。
# 因此这里不主动重装 torch，只安装 OpenMMLab 依赖和当前仓库。
python -m pip install -U pip setuptools wheel openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0,<4.0.0"
python -m pip install -r requirements/runtime.txt
python -m pip install -r requirements/albu.txt || true
python -m pip install -v -e .

echo "[DONE] install finished"
echo "[DONE] next step: put your checkpoint into ./weights/ and run ./run_video_infer.sh"

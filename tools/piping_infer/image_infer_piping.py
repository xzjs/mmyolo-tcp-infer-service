#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
管道缺陷图片/图片文件夹推理脚本。

用途：
1. 保留对“图片入口”的兼容；
2. 在单图或文件夹推理时，同时导出可视化结果与 JSON；
3. 如果模型带有 grading / level 分支，则额外写出 pred_level。
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.utils import mkdir_or_exist

from mmyolo.registry import VISUALIZERS
from mmyolo.utils.misc import get_file_list


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
PIPING_SRC = REPO_ROOT / 'projects' / 'piping' / 'src'

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PIPING_SRC) not in sys.path:
    sys.path.insert(0, str(PIPING_SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Piping image inference script')
    parser.add_argument('img', help='输入图片路径、图片文件夹路径或 URL')
    parser.add_argument('config', help='模型配置文件路径')
    parser.add_argument('checkpoint', help='模型权重路径')
    parser.add_argument('--out-dir', default='./output/images', help='可视化输出目录')
    parser.add_argument('--out-jsonl', default='./output/images/predictions.jsonl', help='预测 JSONL 输出路径')
    parser.add_argument('--device', default='cuda:0', help='推理设备')
    parser.add_argument('--score-thr', type=float, default=0.3, help='检测框置信度阈值')
    return parser.parse_args()


def get_pred_level(data_sample: Any) -> List[Tuple[int, int]]:
    if hasattr(data_sample, 'metainfo') and isinstance(data_sample.metainfo, dict):
        return data_sample.metainfo.get('pred_level', []) or []
    if hasattr(data_sample, '_metainfo') and isinstance(data_sample._metainfo, dict):
        return data_sample._metainfo.get('pred_level', []) or []
    return []


def parse_detection_result(data_sample: Any, score_thr: float, class_names: List[str]) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if not hasattr(data_sample, 'pred_instances'):
        return detections
    pred_instances = data_sample.pred_instances
    if not hasattr(pred_instances, 'scores'):
        return detections

    scores = pred_instances.scores.detach().cpu().numpy()
    labels = pred_instances.labels.detach().cpu().numpy()
    bboxes = pred_instances.bboxes.detach().cpu().numpy()
    for score, label, bbox in zip(scores, labels, bboxes):
        if float(score) < score_thr:
            continue
        label_id = int(label)
        label_name = class_names[label_id] if 0 <= label_id < len(class_names) else str(label_id)
        detections.append({
            'label_id': label_id,
            'label_name': label_name,
            'score': round(float(score), 6),
            'bbox_xyxy': [round(float(v), 2) for v in bbox.tolist()],
        })
    return detections


def parse_grading_result(pred_level: List[Tuple[int, int]], class_names: List[str]) -> List[Dict[str, Any]]:
    grading: List[Dict[str, Any]] = []
    for item in pred_level:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        cls_id = int(item[0])
        grade = int(item[1])
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        grading.append({'class_id': cls_id, 'class_name': cls_name, 'grade': grade})
    return grading


def main() -> None:
    args = parse_args()

    mkdir_or_exist(args.out_dir)
    mkdir_or_exist(os.path.dirname(os.path.abspath(args.out_jsonl)))

    model = init_detector(args.config, args.checkpoint, device=args.device)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    class_names = list(model.dataset_meta.get('classes', []))

    files, source_type = get_file_list(args.img)

    with open(args.out_jsonl, 'w', encoding='utf-8') as f_jsonl:
        for file in files:
            result = inference_detector(model, file)
            detections = parse_detection_result(result, args.score_thr, class_names)
            pred_level = get_pred_level(result)
            grading = parse_grading_result(pred_level, class_names)

            img = mmcv.imread(file)
            img = mmcv.imconvert(img, 'bgr', 'rgb')

            if source_type['is_dir']:
                filename = os.path.relpath(file, args.img).replace('/', '_')
            else:
                filename = os.path.basename(file)
            out_file = os.path.join(args.out_dir, filename)

            visualizer.add_datasample(
                filename,
                img,
                data_sample=result,
                draw_gt=False,
                show=False,
                out_file=out_file,
                pred_score_thr=args.score_thr,
            )

            record = {
                'image': file,
                'detections': detections,
                'pred_level': grading,
            }
            f_jsonl.write(json.dumps(record, ensure_ascii=False) + '\n')

    print('[DONE] image inference finished')
    print(f'[DONE] out_dir   = {args.out_dir}')
    print(f'[DONE] out_jsonl = {args.out_jsonl}')


if __name__ == '__main__':
    main()

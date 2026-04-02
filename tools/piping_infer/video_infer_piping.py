#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
管道缺陷视频推理脚本（带跨帧去重与缺陷图片导出）。

本脚本在原始“逐推理帧输出 jsonl + 输出可视化视频”的基础上，新增了：
1. 跨帧去重：把连续帧里属于同一个缺陷的检测结果合并成一个 defect；
2. 缺陷级结果导出：输出 defects.json；
3. 缺陷图片导出：输出每个缺陷去重后保留的代表整帧图（带框）；
4. 图片路径索引：输出 frame_index.json，记录每张导出整帧图与 defect 的对应关系；
5. 所有新增逻辑都带有中文注释，方便二次维护。

注意：
- 这里并没有“真的把输出 mp4 再解码一遍抽帧”，而是在推理过程中直接缓存最佳帧，
  最终导出代表图。这样质量更好，也更省一次解码开销。
- 仍然保留原始逐推理帧 jsonl，便于问题追溯。
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import mmcv
import numpy as np
import inspect
import torch
from mmcv.transforms import Compose
from mmengine.utils import mkdir_or_exist


def enable_legacy_checkpoint_loading() -> None:
    """兼容 PyTorch 2.6+ 默认 weights_only=True 导致的旧权重加载失败。"""
    if getattr(torch, '_mmyolo_torch_load_patched', False):
        return

    try:
        has_weights_only = 'weights_only' in inspect.signature(torch.load).parameters
    except Exception:
        has_weights_only = False

    original_torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        if has_weights_only and 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)

    torch.load = _torch_load_compat
    torch._mmyolo_torch_load_patched = True

    # 兼容部分旧 checkpoint 中的历史缓冲对象。
    try:
        add_safe_globals = getattr(torch.serialization, 'add_safe_globals', None)
        if callable(add_safe_globals):
            from mmengine.logging.history_buffer import HistoryBuffer
            add_safe_globals([HistoryBuffer])
    except Exception:
        pass


enable_legacy_checkpoint_loading()
from mmdet.apis import inference_detector, init_detector

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:  # pragma: no cover - Pillow 缺失时退化到 OpenCV 文本
    Image = ImageDraw = ImageFont = None  # type: ignore
    PIL_AVAILABLE = False


# =========================================================
# 默认参数
# =========================================================
DEFAULT_INFER_STRIDE = 3
DEFAULT_VIS_STRIDE = 3
DEFAULT_WRITE_STRIDE = 3
DEFAULT_RESIZE_WIDTH = 1280
DEFAULT_RESIZE_HEIGHT = 720
DEFAULT_VIS_SCORE_THR = 0.18
DEFAULT_VIS_TOPK = 10
DEFAULT_DRAW_LABEL_TEXT = True
DEFAULT_DRAW_TRACK_ID = True
DEFAULT_BOX_THICKNESS = 2
DEFAULT_FONT_SIZE = 20
DEFAULT_FONT_PATH = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
DEFAULT_DRAW_GRADE_SUMMARY = True

# 跨帧去重默认参数
DEFAULT_MATCH_IOU_THR = 0.35
DEFAULT_MATCH_CENTER_DIST_RATIO = 0.22
DEFAULT_MAX_FRAME_GAP = 9
DEFAULT_POST_MERGE_GAP_FRAMES = 9
DEFAULT_POST_MERGE_IOU_THR = 0.30
DEFAULT_CHAIN_GAP_FRAMES = 30
DEFAULT_CHAIN_IOU_THR = 0.12
DEFAULT_CHAIN_CENTER_DIST_RATIO = 0.35
DEFAULT_CHAIN_MAX_KEEP = 1
DEFAULT_MIN_HITS = 2
DEFAULT_SINGLE_HIT_HIGH_SCORE = 0.45
DEFAULT_MIN_CROP_SIZE = 8


# =========================================================
# 路径准备
# =========================================================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
PIPING_SRC = REPO_ROOT / 'projects' / 'piping' / 'src'

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PIPING_SRC) not in sys.path:
    sys.path.insert(0, str(PIPING_SRC))


@dataclass
class Track:
    """表示一个跨帧缺陷轨迹。

    这里的 track 是“同一个缺陷在连续若干帧中的出现集合”。
    最终会把 track 再整理成 defect 级结果。
    """

    track_id: int
    label_id: int
    label_name: str
    start_frame: int
    end_frame: int
    first_time_sec: float
    last_time_sec: float
    first_bbox: List[float]
    last_bbox: List[float]
    hit_frames: List[int] = field(default_factory=list)
    hit_count: int = 0
    score_sum: float = 0.0
    best_score: float = -1.0
    best_frame: int = -1
    best_time_sec: float = -1.0
    best_bbox: List[float] = field(default_factory=list)
    best_frame_bgr: Optional[np.ndarray] = None
    all_scores: List[float] = field(default_factory=list)

    def average_score(self) -> float:
        return self.score_sum / max(self.hit_count, 1)


@dataclass
class ExportedDefect:
    """最终导出的 defect 结构。

    注意：当前版本只保留“完整整帧带框图”，不再导出 crop 图。
    """

    defect_id: str
    label_id: int
    label_name: str
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    hit_count: int
    hit_frames: List[int]
    average_score: float
    best_score: float
    best_frame: int
    best_time_sec: float
    best_bbox_xyxy: List[float]
    frame_image_path: str
    track_ids: List[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Piping video inference with cross-frame dedup')
    parser.add_argument('video', help='输入视频路径')
    parser.add_argument('config', help='模型配置文件路径')
    parser.add_argument('checkpoint', help='模型权重路径')

    # 基础推理参数
    parser.add_argument('--device', default='cuda:0', help='推理设备，例如 cuda:0 或 cpu')
    parser.add_argument('--score-thr', type=float, default=0.3, help='检测框保底阈值')
    parser.add_argument('--out-video', required=True, help='输出可视化视频路径')
    parser.add_argument('--out-jsonl', required=True, help='输出逐推理帧 JSONL 路径')
    parser.add_argument('--out-defects-json', default='', help='输出缺陷级 JSON 路径；为空则按 out_jsonl 自动推导')
    parser.add_argument('--out-frame-index-json', default='', help='输出图片路径索引 JSON 路径；为空则按 out_jsonl 自动推导')
    parser.add_argument('--save-defect-frames-dir', default='', help='导出每个缺陷代表整帧图的目录；为空则按 out_jsonl 自动推导')
    parser.add_argument('--save-defect-crops-dir', default='', help='兼容旧参数：当前版本不再导出裁剪图，可留空')
    parser.add_argument('--save-vis-frames-dir', default='', help='可选，额外保存逐帧可视化图片目录')
    parser.add_argument('--show', action='store_true', help='是否实时弹窗显示结果')
    parser.add_argument('--wait-time', type=int, default=1, help='显示窗口等待时间，单位毫秒')
    parser.add_argument('--max-frames', type=int, default=-1, help='最多处理多少源视频帧，-1 表示全部处理')
    parser.add_argument('--print-every', type=int, default=30, help='每隔多少源视频帧打印一次进度')

    # 抽帧 / 输出控制
    parser.add_argument('--infer-stride', type=int, default=DEFAULT_INFER_STRIDE, help='每多少帧做一次推理')
    parser.add_argument('--vis-stride', type=int, default=DEFAULT_VIS_STRIDE, help='每多少帧做一次可视化')
    parser.add_argument('--write-stride', type=int, default=DEFAULT_WRITE_STRIDE, help='每多少帧写一次输出视频')
    parser.add_argument('--resize-width', type=int, default=DEFAULT_RESIZE_WIDTH, help='推理/输出统一宽度')
    parser.add_argument('--resize-height', type=int, default=DEFAULT_RESIZE_HEIGHT, help='推理/输出统一高度')
    parser.add_argument('--vis-score-thr', type=float, default=DEFAULT_VIS_SCORE_THR, help='可视化与缺陷级保留的最低分数阈值')
    parser.add_argument('--vis-topk', type=int, default=DEFAULT_VIS_TOPK, help='每个推理帧最多保留多少个检测框')
    parser.add_argument('--font-path', default=DEFAULT_FONT_PATH, help='中文字体路径')
    parser.add_argument('--font-size', type=int, default=DEFAULT_FONT_SIZE, help='中文字体大小')
    parser.add_argument('--draw-label-text', action='store_true', default=DEFAULT_DRAW_LABEL_TEXT, help='是否绘制类别名与分数')
    parser.add_argument('--no-draw-label-text', dest='draw_label_text', action='store_false', help='关闭类别名绘制')
    parser.add_argument('--draw-track-id', action='store_true', default=DEFAULT_DRAW_TRACK_ID, help='是否绘制临时轨迹 ID')
    parser.add_argument('--no-draw-track-id', dest='draw_track_id', action='store_false', help='关闭临时轨迹 ID 绘制')
    parser.add_argument('--draw-grade-summary', action='store_true', default=DEFAULT_DRAW_GRADE_SUMMARY, help='是否绘制左上角 pred_level 摘要')
    parser.add_argument('--no-draw-grade-summary', dest='draw_grade_summary', action='store_false', help='关闭 pred_level 摘要绘制')

    # 去重逻辑参数
    parser.add_argument('--match-iou-thr', type=float, default=DEFAULT_MATCH_IOU_THR, help='跨帧关联时，同类框最小 IoU')
    parser.add_argument('--match-center-dist-ratio', type=float, default=DEFAULT_MATCH_CENTER_DIST_RATIO,
                        help='跨帧关联时，框中心距离 / 画面对角线 的最大比例')
    parser.add_argument('--max-frame-gap', type=int, default=DEFAULT_MAX_FRAME_GAP,
                        help='轨迹允许中断的最大帧间隔；超过后会 finalize')
    parser.add_argument('--post-merge-gap-frames', type=int, default=DEFAULT_POST_MERGE_GAP_FRAMES,
                        help='离线二次合并时，两个 track 允许相隔的最大帧数')
    parser.add_argument('--post-merge-iou-thr', type=float, default=DEFAULT_POST_MERGE_IOU_THR,
                        help='离线二次合并时，前后两个 track 的最小 IoU')
    parser.add_argument('--chain-gap-frames', type=int, default=DEFAULT_CHAIN_GAP_FRAMES,
                        help='缺陷链聚合时，前后两个 track 允许相隔的最大帧数；用于进一步压缩重复缺陷')
    parser.add_argument('--chain-iou-thr', type=float, default=DEFAULT_CHAIN_IOU_THR,
                        help='缺陷链聚合时，前后两个 track 的最小 IoU；越小越容易被认为是同一串缺陷')
    parser.add_argument('--chain-center-dist-ratio', type=float, default=DEFAULT_CHAIN_CENTER_DIST_RATIO,
                        help='缺陷链聚合时，框中心距离 / 画面对角线 的最大比例')
    parser.add_argument('--chain-max-keep', type=int, default=DEFAULT_CHAIN_MAX_KEEP,
                        help='同一串重复缺陷最终最多保留多少个代表样本；当前默认 1，表示优先只保留第一个')
    parser.add_argument('--min-hits', type=int, default=DEFAULT_MIN_HITS,
                        help='一个缺陷至少出现多少次才保留')
    parser.add_argument('--single-hit-high-score', type=float, default=DEFAULT_SINGLE_HIT_HIGH_SCORE,
                        help='单次命中的缺陷如果分数高于该阈值，也允许保留')
    parser.add_argument('--min-crop-size', type=int, default=DEFAULT_MIN_CROP_SIZE,
                        help='兼容旧参数：当前版本不再导出裁剪图，此参数仅保留兼容性')

    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        mkdir_or_exist(parent)


def get_derived_output_paths(args: argparse.Namespace) -> Tuple[str, str, str, str, str]:
    """按 out_jsonl 的前缀自动推导新增产物路径。

    说明：
    - 当前版本所有带框缺陷整帧图统一落到一个目录里；
    - 不再导出 crop 图，但为了兼容旧 TCP/脚本参数，仍然返回 save_defect_crops_dir。
    """
    stem = os.path.splitext(args.out_jsonl)[0]
    out_defects_json = args.out_defects_json or f'{stem}_defects.json'
    out_frame_index_json = args.out_frame_index_json or f'{stem}_frame_index.json'
    save_defect_frames_dir = args.save_defect_frames_dir or f'{stem}_defect_images'
    save_defect_crops_dir = args.save_defect_crops_dir or ''
    class_map_path = f'{stem}_classes.json'
    return out_defects_json, out_frame_index_json, save_defect_frames_dir, save_defect_crops_dir, class_map_path


def get_pred_level(data_sample: Any) -> List[Tuple[int, int]]:
    if hasattr(data_sample, 'metainfo') and isinstance(data_sample.metainfo, dict):
        return data_sample.metainfo.get('pred_level', []) or []
    if hasattr(data_sample, '_metainfo') and isinstance(data_sample._metainfo, dict):
        return data_sample._metainfo.get('pred_level', []) or []
    try:
        meta = data_sample.metainfo_keys()
        if 'pred_level' in meta:
            return data_sample.get('pred_level', []) or []
    except Exception:
        pass
    return []


def to_python_number(x: Any) -> Any:
    if hasattr(x, 'item'):
        try:
            return x.item()
        except Exception:
            return x
    return x


def parse_detection_result(
    data_sample: Any,
    score_thr: float,
    class_names: List[str],
    topk: int,
) -> List[Dict[str, Any]]:
    """解析检测结果，只保留高分前 topk 个框。"""
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
        score_f = float(score)
        if score_f < score_thr:
            continue

        label_id = int(label)
        label_name = class_names[label_id] if 0 <= label_id < len(class_names) else str(label_id)
        detections.append({
            'label_id': label_id,
            'label_name': label_name,
            'score': round(score_f, 6),
            'bbox_xyxy': [round(float(v), 2) for v in bbox.tolist()],
        })

    detections.sort(key=lambda x: x['score'], reverse=True)
    return detections[:max(topk, 0)]


def parse_grading_result(pred_level: List[Tuple[int, int]], class_names: List[str]) -> List[Dict[str, Any]]:
    grading: List[Dict[str, Any]] = []
    for item in pred_level:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        cls_id = int(to_python_number(item[0]))
        grade = int(to_python_number(item[1]))
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        grading.append({
            'class_id': cls_id,
            'class_name': cls_name,
            'grade': grade,
        })
    return grading


def get_pil_font(font_path: str, size: int):
    if not PIL_AVAILABLE:
        return None
    candidate_fonts = [
        font_path,
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for one_path in candidate_fonts:
        try:
            if one_path and os.path.isfile(one_path):
                return ImageFont.truetype(one_path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_text_pil(
    frame_bgr: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_path: str,
    font_size: int,
    text_color=(0, 0, 0),
    bg_color=(0, 255, 0),
) -> np.ndarray:
    """用 PIL 在 BGR 图像上绘制中文文本；若 Pillow 不可用则退回 OpenCV。"""
    if not PIL_AVAILABLE:
        cv2.putText(frame_bgr, text.encode('ascii', errors='replace').decode('ascii'), (x, max(y + font_size, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, bg_color, 2, cv2.LINE_AA)
        return frame_bgr

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = get_pil_font(font_path, font_size)
    text_to_draw = text

    def _safe_text(raw_text: str) -> str:
        try:
            raw_text.encode('latin-1')
            return raw_text
        except UnicodeEncodeError:
            return raw_text.encode('latin-1', errors='replace').decode('latin-1')

    try:
        x1, y1, x2, y2 = draw.textbbox((x, y), text_to_draw, font=font)
    except Exception:
        text_to_draw = _safe_text(text_to_draw)
        try:
            text_w, text_h = draw.textsize(text_to_draw, font=font)
        except Exception:
            text_w, text_h = font.getsize(text_to_draw) if font is not None else (len(text_to_draw) * font_size, font_size)
        x1, y1 = x, y
        x2, y2 = x + text_w, y + text_h

    pad = 3
    draw.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad], fill=bg_color)
    try:
        draw.text((x, y), text_to_draw, font=font, fill=text_color)
    except Exception:
        draw.text((x, y), _safe_text(text_to_draw), font=font, fill=text_color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_grade_summary(frame: np.ndarray, grading: List[Dict[str, Any]], enabled: bool) -> np.ndarray:
    if not enabled or not grading:
        return frame

    overlay_lines = ['frame-level grades:']
    for item in grading:
        overlay_lines.append(f"cls_{item['class_id']} -> grade_{item['grade']}")

    y = 28
    for line in overlay_lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return frame


def draw_simple_bboxes(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    draw_label_text: bool,
    draw_track_id: bool,
    box_thickness: int,
    font_path: str,
    font_size: int,
) -> np.ndarray:
    """绘制 bbox、类别名和临时轨迹 ID。"""
    h, w = frame.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox_xyxy']]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

        if draw_label_text:
            text = f"{det['label_name']} {det['score']:.2f}"
            if draw_track_id and 'track_id' in det:
                text += f"  T{det['track_id']}"
            text_y = max(y1 - font_size - 6, 0)
            frame = draw_text_pil(frame, text, x1, text_y, font_path, font_size)

    return frame


def bbox_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(float(x2) - float(x1), 0.0) * max(float(y2) - float(y1), 0.0)


def bbox_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    x11, y11, x12, y12 = [float(v) for v in box1]
    x21, y21, x22, y22 = [float(v) for v in box2]

    inter_x1 = max(x11, x21)
    inter_y1 = max(y11, y21)
    inter_x2 = min(x12, x22)
    inter_y2 = min(y12, y22)
    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h
    union = bbox_area(box1) + bbox_area(box2) - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def bbox_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_center_distance(box1: Sequence[float], box2: Sequence[float]) -> float:
    """计算两个检测框中心点之间的欧氏距离。

    这个函数是链式去重逻辑会直接调用的基础函数。
    之前报错 `NameError: bbox_center_distance is not defined`，
    就是因为这里漏定义了。当前版本已补齐。
    """
    c1x, c1y = bbox_center(box1)
    c2x, c2y = bbox_center(box2)
    return math.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2)


def center_distance_ratio(box1: Sequence[float], box2: Sequence[float], frame_shape: Tuple[int, int, int]) -> float:
    """框中心距离 / 画面对角线。

    这个条件专门用来抑制“同类但位置不同”的误合并。
    """
    h, w = frame_shape[:2]
    diag = math.sqrt(float(w * w + h * h))
    if diag <= 1e-6:
        return 1.0
    c1x, c1y = bbox_center(box1)
    c2x, c2y = bbox_center(box2)
    dist = math.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2)
    return dist / diag


def clamp_bbox(box: Sequence[float], frame_shape: Tuple[int, int, int]) -> List[int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return [x1, y1, x2, y2]


def expand_tiny_bbox(box: Sequence[float], frame_shape: Tuple[int, int, int], min_crop_size: int) -> List[int]:
    """如果框过小，做一个最小裁剪扩张，避免导出的 crop 太小看不清。"""
    x1, y1, x2, y2 = clamp_bbox(box, frame_shape)
    h, w = frame_shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    if bw >= min_crop_size and bh >= min_crop_size:
        return [x1, y1, x2, y2]

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = max(min_crop_size // 2, 1)
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w - 1)
    y2 = min(cy + half, h - 1)
    return [x1, y1, x2, y2]


def create_track(track_id: int, det: Dict[str, Any], frame_idx: int, time_sec: float, frame_bgr: np.ndarray) -> Track:
    box = [float(v) for v in det['bbox_xyxy']]
    score = float(det['score'])
    return Track(
        track_id=track_id,
        label_id=int(det['label_id']),
        label_name=str(det['label_name']),
        start_frame=frame_idx,
        end_frame=frame_idx,
        first_time_sec=time_sec,
        last_time_sec=time_sec,
        first_bbox=box.copy(),
        last_bbox=box.copy(),
        hit_frames=[frame_idx],
        hit_count=1,
        score_sum=score,
        best_score=score,
        best_frame=frame_idx,
        best_time_sec=time_sec,
        best_bbox=box.copy(),
        best_frame_bgr=frame_bgr.copy(),
        all_scores=[score],
    )


def update_track(track: Track, det: Dict[str, Any], frame_idx: int, time_sec: float, frame_bgr: np.ndarray) -> None:
    box = [float(v) for v in det['bbox_xyxy']]
    score = float(det['score'])
    track.end_frame = frame_idx
    track.last_time_sec = time_sec
    track.last_bbox = box.copy()
    track.hit_frames.append(frame_idx)
    track.hit_count += 1
    track.score_sum += score
    track.all_scores.append(score)
    if score >= track.best_score:
        track.best_score = score
        track.best_frame = frame_idx
        track.best_time_sec = time_sec
        track.best_bbox = box.copy()
        track.best_frame_bgr = frame_bgr.copy()


def compute_match_score(
    track: Track,
    det: Dict[str, Any],
    frame_idx: int,
    frame_shape: Tuple[int, int, int],
    args: argparse.Namespace,
) -> Tuple[bool, float, Dict[str, float]]:
    """判断 det 是否能匹配到现有 track。

    这里同时使用：
    - 同类约束；
    - 时间间隔约束；
    - IoU 约束；
    - 框中心距离约束；

    最终 score 采用 “IoU 越高越好，中心距离越小越好”。
    """
    if int(det['label_id']) != track.label_id:
        return False, -1.0, {}

    gap = frame_idx - track.end_frame
    if gap <= 0 or gap > max(int(args.max_frame_gap), 1):
        return False, -1.0, {}

    cur_box = det['bbox_xyxy']
    iou = bbox_iou(track.last_bbox, cur_box)
    if iou < float(args.match_iou_thr):
        return False, -1.0, {'iou': iou}

    cdist_ratio = center_distance_ratio(track.last_bbox, cur_box, frame_shape)
    if cdist_ratio > float(args.match_center_dist_ratio):
        return False, -1.0, {'iou': iou, 'center_dist_ratio': cdist_ratio}

    # 匹配分数：更偏向 IoU，同时轻度惩罚中心偏移。
    score = iou - 0.35 * cdist_ratio
    return True, score, {'iou': iou, 'center_dist_ratio': cdist_ratio, 'gap': float(gap)}


def finalize_stale_tracks(active_tracks: List[Track], frame_idx: int, max_frame_gap: int) -> Tuple[List[Track], List[Track]]:
    """把超时未更新的轨迹从活动列表中拿出来，转入 finalized。"""
    still_active: List[Track] = []
    finalized: List[Track] = []
    for track in active_tracks:
        if frame_idx - track.end_frame > max(max_frame_gap, 1):
            finalized.append(track)
        else:
            still_active.append(track)
    return still_active, finalized


def should_keep_track(track: Track, args: argparse.Namespace) -> bool:
    """决定一个轨迹最终是否导出为缺陷。"""
    if track.hit_count >= max(int(args.min_hits), 1):
        return True
    if track.best_score >= float(args.single_hit_high_score):
        return True
    return False


def can_post_merge(prev_track: Track, cur_track: Track, frame_shape: Tuple[int, int, int], args: argparse.Namespace) -> bool:
    """离线二次合并：用于解决轨迹被短暂打断后拆成两个 defect 的问题。"""
    if prev_track.label_id != cur_track.label_id:
        return False
    gap = cur_track.start_frame - prev_track.end_frame
    if gap < 0 or gap > max(int(args.post_merge_gap_frames), 0):
        return False
    iou = bbox_iou(prev_track.last_bbox, cur_track.first_bbox)
    if iou < float(args.post_merge_iou_thr):
        return False
    cdist_ratio = center_distance_ratio(prev_track.last_bbox, cur_track.first_bbox, frame_shape)
    if cdist_ratio > float(args.match_center_dist_ratio) * 1.2:
        return False
    return True


def merge_two_tracks(base_track: Track, new_track: Track) -> Track:
    """把 new_track 合并进 base_track。"""
    base_track.end_frame = max(base_track.end_frame, new_track.end_frame)
    base_track.last_time_sec = max(base_track.last_time_sec, new_track.last_time_sec)
    base_track.last_bbox = new_track.last_bbox.copy()
    base_track.hit_frames.extend(new_track.hit_frames)
    base_track.hit_frames = sorted(set(base_track.hit_frames))
    base_track.hit_count = len(base_track.hit_frames)
    base_track.score_sum += new_track.score_sum
    base_track.all_scores.extend(new_track.all_scores)
    if new_track.best_score >= base_track.best_score:
        base_track.best_score = new_track.best_score
        base_track.best_frame = new_track.best_frame
        base_track.best_time_sec = new_track.best_time_sec
        base_track.best_bbox = new_track.best_bbox.copy()
        base_track.best_frame_bgr = None if new_track.best_frame_bgr is None else new_track.best_frame_bgr.copy()
    return base_track


def post_merge_tracks(finalized_tracks: List[Track], frame_shape: Tuple[int, int, int], args: argparse.Namespace) -> List[Track]:
    """离线二次合并 finalized 结果。

    这样可以修补少量漏检导致的一次断轨。
    """
    if not finalized_tracks:
        return []

    tracks = sorted(finalized_tracks, key=lambda t: (t.label_id, t.start_frame, t.track_id))
    merged: List[Track] = []
    for track in tracks:
        if not merged:
            merged.append(track)
            continue
        prev = merged[-1]
        if can_post_merge(prev, track, frame_shape, args):
            merged[-1] = merge_two_tracks(prev, track)
        else:
            merged.append(track)
    return merged


def can_chain_merge(prev_track: Track, cur_track: Track, frame_shape: Tuple[int, int, int], args: argparse.Namespace) -> bool:
    """判断两个已经保留的 track 是否仍然属于同一串重复缺陷。

    这一步比 post-merge 更激进：
    - post-merge 主要修补“断轨”；
    - chain-merge 主要压缩“连续很多次都像同一个缺陷”的重复输出。
    """
    if prev_track.label_id != cur_track.label_id:
        return False

    gap = int(cur_track.start_frame) - int(prev_track.end_frame)
    if gap < 0 or gap > max(int(args.chain_gap_frames), 0):
        return False

    iou = bbox_iou(prev_track.best_bbox, cur_track.best_bbox)
    if iou >= float(args.chain_iou_thr):
        return True

    diag = math.sqrt(float(frame_shape[0] ** 2 + frame_shape[1] ** 2))
    center_dist = bbox_center_distance(prev_track.best_bbox, cur_track.best_bbox)
    return center_dist <= float(args.chain_center_dist_ratio) * max(diag, 1.0)


def build_defect_groups(kept_tracks: List[Track], frame_shape: Tuple[int, int, int], args: argparse.Namespace) -> List[List[Track]]:
    """把仍然疑似重复的一串 track 聚成一个 defect group。

    这里特别按"视频时间顺序"来组织输出，而不是按类别分桶。
    这样最终导出的图片文件名、frame_index.json、defects.json 都会尽量贴近
    原视频中的出现顺序，便于人工回看。

    做法：
    1. 先按 start_frame 排序；
    2. 仅与"当前组最后一个 track"比较是否可链式合并；
    3. 只要类别不同，或时间/位置不满足，就立刻开新组。
    """
    if not kept_tracks:
        return []

    tracks = sorted(kept_tracks, key=lambda t: (t.start_frame, t.best_frame, t.track_id))
    groups: List[List[Track]] = []
    cur_group: List[Track] = []

    for track in tracks:
        if not cur_group:
            cur_group = [track]
            continue

        prev = cur_group[-1]
        if can_chain_merge(prev, track, frame_shape, args):
            cur_group.append(track)
        else:
            groups.append(cur_group)
            cur_group = [track]

    if cur_group:
        groups.append(cur_group)

    # 再做一次组级别排序，确保最终 defect_id 也是按视频顺序编号。
    groups = sorted(
        groups,
        key=lambda g: (
            min(int(t.start_frame) for t in g),
            min(int(t.best_frame) for t in g),
            min(int(t.track_id) for t in g),
        ),
    )
    return groups


def sample_group_tracks(group_tracks: List[Track], max_keep: int) -> List[Track]:
    """从同一串重复缺陷里抽代表样本。

    规则：
    - 数量不多：全保留；
    - 数量很多：均匀抽样，但一定保留第一个和最后一个。
    """
    if not group_tracks:
        return []

    max_keep = max(int(max_keep), 1)
    if len(group_tracks) <= max_keep:
        return list(group_tracks)

    if max_keep == 1:
        return [group_tracks[0]]

    n = len(group_tracks)
    indices = set()
    for i in range(max_keep):
        idx = round(i * (n - 1) / (max_keep - 1))
        indices.add(int(idx))
    return [group_tracks[i] for i in sorted(indices)]


def aggregate_group_tracks(group_tracks: List[Track]) -> Dict[str, Any]:
    """把一个重复缺陷组汇总成 defect 级统计信息。"""
    assert group_tracks, 'group_tracks cannot be empty'
    best_track = max(group_tracks, key=lambda t: float(t.best_score))
    hit_frames: List[int] = []
    track_ids: List[int] = []
    score_sum = 0.0
    hit_count = 0
    for track in group_tracks:
        hit_frames.extend(track.hit_frames)
        track_ids.append(track.track_id)
        score_sum += float(track.score_sum)
        hit_count += int(track.hit_count)
    hit_frames = sorted(set(int(v) for v in hit_frames))
    return {
        'label_id': best_track.label_id,
        'label_name': best_track.label_name,
        'start_frame': min(int(t.start_frame) for t in group_tracks),
        'end_frame': max(int(t.end_frame) for t in group_tracks),
        'start_time_sec': float(min(t.first_time_sec for t in group_tracks)),
        'end_time_sec': float(max(t.last_time_sec for t in group_tracks)),
        'hit_count': hit_count,
        'hit_frames': hit_frames,
        'average_score': score_sum / max(hit_count, 1),
        'best_score': float(best_track.best_score),
        'best_frame': int(best_track.best_frame),
        'best_time_sec': float(best_track.best_time_sec),
        'best_bbox_xyxy': list(best_track.best_bbox),
        'track_ids': track_ids,
        'group_size': len(group_tracks),
        'sampled_track_ids': [],
    }


def draw_box_on_crop(crop_bgr: np.ndarray, original_box: Sequence[float], crop_xyxy: Sequence[int], label_text: str, args: argparse.Namespace) -> np.ndarray:
    """在裁剪图上把原框再画出来，避免导出“无框 crop”。"""
    x1, y1, x2, y2 = [int(round(v)) for v in crop_xyxy]
    bx1, by1, bx2, by2 = [float(v) for v in original_box]
    rel_box = [max(0.0, bx1 - x1), max(0.0, by1 - y1), max(0.0, bx2 - x1), max(0.0, by2 - y1)]
    draw_det = {
        'label_id': -1,
        'label_name': label_text,
        'score': 0.0,
        'bbox_xyxy': rel_box,
        'track_id': None,
    }
    return draw_simple_bboxes(
        crop_bgr,
        [draw_det],
        draw_label_text=True,
        draw_track_id=False,
        box_thickness=DEFAULT_BOX_THICKNESS,
        font_path=args.font_path,
        font_size=int(args.font_size),
    )


def save_defect_artifacts(
    kept_tracks: List[Track],
    save_defect_frames_dir: str,
    save_defect_crops_dir: str,
    frame_index_json_path: str,
    defects_json_path: str,
    frame_shape: Tuple[int, int, int],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """把每个保留缺陷的代表整帧图（带框）和索引 JSON 输出到磁盘。

    当前版本按你的新要求做了这几件事：
    1. 修复链式去重依赖函数缺失的问题；
    2. 所有导出的缺陷图片都保存为“完整整帧带框图”；
    3. 所有图片统一放到一个目录里，不再按 defect 建子目录；
    4. 不再导出 crop 图；
    5. 单独输出一个 JSON，记录每张带框整帧图与 defect 的对应关系。

    说明：
    - 参数 save_defect_crops_dir 仅为兼容旧接口保留，当前版本不使用；
    - frame_index_json_path 里只记录整帧图片路径，不再记录 crop 路径。
    """
    mkdir_or_exist(save_defect_frames_dir)
    ensure_parent(frame_index_json_path)
    ensure_parent(defects_json_path)

    frame_index_records: List[Dict[str, Any]] = []
    exported_defects: List[Dict[str, Any]] = []

    defect_groups = build_defect_groups(kept_tracks, frame_shape, args)

    for idx, group_tracks in enumerate(defect_groups, start=1):
        defect_id = f'defect_{idx:06d}'
        defect_summary = aggregate_group_tracks(group_tracks)
        sampled_tracks = sample_group_tracks(group_tracks, int(args.chain_max_keep))
        # 样本图也按视频先后排序，避免同一 defect 内部图片乱序。
        sampled_tracks = sorted(sampled_tracks, key=lambda t: (t.best_frame, t.start_frame, t.track_id))
        defect_summary['sampled_track_ids'] = [int(t.track_id) for t in sampled_tracks]

        sample_records: List[Dict[str, Any]] = []

        for sample_idx, track in enumerate(sampled_tracks, start=1):
            best_frame_bgr = track.best_frame_bgr.copy() if track.best_frame_bgr is not None else None
            if best_frame_bgr is None:
                continue

            image_name = f'{defect_id}_sample_{sample_idx:02d}_frame_{track.best_frame:06d}.jpg'
            image_out_path = os.path.join(save_defect_frames_dir, image_name)

            draw_det = {
                'label_id': track.label_id,
                'label_name': track.label_name,
                'score': round(float(track.best_score), 6),
                'bbox_xyxy': [round(float(v), 2) for v in track.best_bbox],
                'track_id': track.track_id,
            }
            vis_frame = draw_simple_bboxes(
                best_frame_bgr.copy(),
                [draw_det],
                draw_label_text=True,
                draw_track_id=True,
                box_thickness=DEFAULT_BOX_THICKNESS,
                font_path=args.font_path,
                font_size=int(args.font_size),
            )
            mmcv.imwrite(vis_frame, image_out_path)

            sample_record = {
                'sample_index': sample_idx,
                'track_id': int(track.track_id),
                'label_id': int(track.label_id),
                'label_name': track.label_name,
                'best_frame': int(track.best_frame),
                'best_time_sec': round(float(track.best_time_sec), 6),
                'best_score': round(float(track.best_score), 6),
                'best_bbox_xyxy': [round(float(v), 2) for v in track.best_bbox],
                'image_path': image_out_path,
            }
            sample_records.append(sample_record)
            frame_index_records.append({'defect_id': defect_id, **sample_record})

        if not sample_records:
            continue

        exported_defects.append({
            'defect_id': defect_id,
            'label_id': int(defect_summary['label_id']),
            'label_name': defect_summary['label_name'],
            'start_frame': int(defect_summary['start_frame']),
            'end_frame': int(defect_summary['end_frame']),
            'start_time_sec': round(float(defect_summary['start_time_sec']), 6),
            'end_time_sec': round(float(defect_summary['end_time_sec']), 6),
            'hit_count': int(defect_summary['hit_count']),
            'hit_frames': list(defect_summary['hit_frames']),
            'average_score': round(float(defect_summary['average_score']), 6),
            'best_score': round(float(defect_summary['best_score']), 6),
            'best_frame': int(defect_summary['best_frame']),
            'best_time_sec': round(float(defect_summary['best_time_sec']), 6),
            'best_bbox_xyxy': [round(float(v), 2) for v in defect_summary['best_bbox_xyxy']],
            'track_ids': list(defect_summary['track_ids']),
            'group_size': int(defect_summary['group_size']),
            'sampled_track_ids': list(defect_summary['sampled_track_ids']),
            'sample_count': len(sample_records),
            'samples': sample_records,
        })

    # 最终落盘前再统一按视频顺序排序一次，保证 JSON 顺序和图片浏览顺序一致。
    frame_index_records = sorted(
        frame_index_records,
        key=lambda x: (int(x['best_frame']), int(x['sample_index']), str(x['defect_id'])),
    )
    exported_defects = sorted(
        exported_defects,
        key=lambda x: (int(x['start_frame']), int(x['best_frame']), str(x['defect_id'])),
    )

    with open(frame_index_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_images': len(frame_index_records),
            'image_dir': save_defect_frames_dir,
            'order_rule': '按视频帧顺序排序（best_frame 从小到大）',
            'note': '当前版本仅输出带框整帧图；items 中 image_path 即最终图片路径。',
            'items': frame_index_records,
        }, f, ensure_ascii=False, indent=2)

    with open(defects_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_defects': len(exported_defects),
            'image_dir': save_defect_frames_dir,
            'order_rule': '按视频帧顺序排序（start_frame / best_frame 从小到大）',
            'items': exported_defects,
        }, f, ensure_ascii=False, indent=2)

    return frame_index_records, exported_defects


def main() -> None:

    args = parse_args()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(f'输入视频不存在: {args.video}')
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f'配置文件不存在: {args.config}')
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f'模型权重不存在: {args.checkpoint}')

    out_defects_json, out_frame_index_json, save_defect_frames_dir, save_defect_crops_dir, class_map_path = get_derived_output_paths(args)

    ensure_parent(args.out_video)
    ensure_parent(args.out_jsonl)
    ensure_parent(out_defects_json)
    ensure_parent(out_frame_index_json)
    ensure_parent(class_map_path)
    mkdir_or_exist(save_defect_frames_dir)
    if save_defect_crops_dir:
        mkdir_or_exist(save_defect_crops_dir)
    if args.save_vis_frames_dir:
        mkdir_or_exist(args.save_vis_frames_dir)

    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    if hasattr(model, 'dataset_meta') and isinstance(model.dataset_meta, dict):
        class_names = list(model.dataset_meta.get('classes', []))
    else:
        class_names = []

    video_reader = mmcv.VideoReader(args.video)
    if len(video_reader) == 0:
        raise RuntimeError(f'无法读取视频内容: {args.video}')

    src_fps = float(video_reader.fps)
    out_fps = max(src_fps / max(int(args.write_stride), 1), 1.0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        args.out_video,
        fourcc,
        out_fps,
        (int(args.resize_width), int(args.resize_height)),
    )
    if not video_writer.isOpened():
        raise RuntimeError(f'输出视频无法写入: {args.out_video}')

    with open(class_map_path, 'w', encoding='utf-8') as f:
        json.dump({
            'classes': class_names,
            'note': 'pred_level 为帧级按类别聚合后的等级结果；defects.json 为跨帧去重后的缺陷级结果。',
        }, f, ensure_ascii=False, indent=2)

    start_time = time.time()
    processed_source_frames = 0
    frame_idx = -1
    frame_shape = (int(args.resize_height), int(args.resize_width), 3)

    last_result = None
    last_vis_frame_bgr: Optional[np.ndarray] = None
    last_detections: List[Dict[str, Any]] = []
    last_grading: List[Dict[str, Any]] = []
    window_created = False

    active_tracks: List[Track] = []
    finalized_tracks: List[Track] = []
    next_track_id = 1

    with open(args.out_jsonl, 'w', encoding='utf-8') as f_jsonl:
        for frame in video_reader:
            frame_idx += 1
            if frame is None:
                continue
            if args.max_frames > 0 and processed_source_frames >= args.max_frames:
                break

            frame = cv2.resize(frame, (int(args.resize_width), int(args.resize_height)))
            frame_shape = frame.shape
            time_sec = round(frame_idx / max(src_fps, 1e-6), 6)

            need_infer = (frame_idx % max(int(args.infer_stride), 1) == 0)
            need_vis = (frame_idx % max(int(args.vis_stride), 1) == 0)
            need_write = (frame_idx % max(int(args.write_stride), 1) == 0)

            # 先把过期轨迹 finalize 掉，避免后续误匹配。
            active_tracks, stale_tracks = finalize_stale_tracks(active_tracks, frame_idx, int(args.max_frame_gap))
            finalized_tracks.extend(stale_tracks)

            if need_infer:
                result = inference_detector(model, frame, test_pipeline=test_pipeline)
                detections = parse_detection_result(result, float(args.vis_score_thr), class_names, int(args.vis_topk))
                pred_level = get_pred_level(result)
                grading = parse_grading_result(pred_level, class_names)

                # 逐个检测框做跨帧关联。
                used_track_ids = set()
                enriched_detections: List[Dict[str, Any]] = []
                detections_sorted = sorted(detections, key=lambda x: x['score'], reverse=True)
                for det in detections_sorted:
                    best_track: Optional[Track] = None
                    best_match_score = -1e9
                    for track in active_tracks:
                        if track.track_id in used_track_ids:
                            continue
                        ok, match_score, _ = compute_match_score(track, det, frame_idx, frame_shape, args)
                        if ok and match_score > best_match_score:
                            best_match_score = match_score
                            best_track = track

                    det_copy = dict(det)
                    if best_track is not None:
                        update_track(best_track, det_copy, frame_idx, time_sec, frame)
                        det_copy['track_id'] = best_track.track_id
                        used_track_ids.add(best_track.track_id)
                    else:
                        new_track = create_track(next_track_id, det_copy, frame_idx, time_sec, frame)
                        active_tracks.append(new_track)
                        det_copy['track_id'] = new_track.track_id
                        used_track_ids.add(new_track.track_id)
                        next_track_id += 1

                    enriched_detections.append(det_copy)

                last_result = result
                last_detections = enriched_detections
                last_grading = grading

                record = {
                    'frame_index': frame_idx,
                    'time_sec': time_sec,
                    'detections': enriched_detections,
                    'pred_level': grading,
                }
                f_jsonl.write(json.dumps(record, ensure_ascii=False) + '\n')

            if need_vis and last_result is not None:
                vis_frame = frame.copy()
                vis_frame = draw_simple_bboxes(
                    vis_frame,
                    last_detections,
                    draw_label_text=bool(args.draw_label_text),
                    draw_track_id=bool(args.draw_track_id),
                    box_thickness=DEFAULT_BOX_THICKNESS,
                    font_path=args.font_path,
                    font_size=int(args.font_size),
                )
                vis_frame = draw_grade_summary(vis_frame, last_grading, enabled=bool(args.draw_grade_summary))
                last_vis_frame_bgr = vis_frame

                if args.save_vis_frames_dir:
                    out_img = os.path.join(args.save_vis_frames_dir, f'frame_{frame_idx:06d}.jpg')
                    mmcv.imwrite(last_vis_frame_bgr, out_img)

                if args.show:
                    if not window_created:
                        try:
                            cv2.namedWindow('piping_video_infer', 0)
                            window_created = True
                        except cv2.error:
                            window_created = False
                    if window_created:
                        try:
                            mmcv.imshow(last_vis_frame_bgr, 'piping_video_infer', args.wait_time)
                        except cv2.error:
                            pass

            if need_write and last_vis_frame_bgr is not None:
                video_writer.write(last_vis_frame_bgr)

            processed_source_frames += 1
            if processed_source_frames % max(int(args.print_every), 1) == 0:
                elapsed = time.time() - start_time
                fps = processed_source_frames / max(elapsed, 1e-6)
                print(f'[INFO] processed={processed_source_frames}, frame_idx={frame_idx}, avg_fps={fps:.2f}')

    # 视频结束后，把剩余活动轨迹全部 finalize。
    finalized_tracks.extend(active_tracks)
    active_tracks = []

    # 二次合并，修补轻微断轨。
    finalized_tracks = post_merge_tracks(finalized_tracks, frame_shape, args)

    # 按保留规则筛掉明显噪声。
    kept_tracks = [track for track in finalized_tracks if should_keep_track(track, args)]
    kept_tracks.sort(key=lambda t: (t.start_frame, t.label_id, t.track_id))

    frame_index_records, exported_defects = save_defect_artifacts(
        kept_tracks=kept_tracks,
        save_defect_frames_dir=save_defect_frames_dir,
        save_defect_crops_dir=save_defect_crops_dir,
        frame_index_json_path=out_frame_index_json,
        defects_json_path=out_defects_json,
        frame_shape=frame_shape,
        args=args,
    )

    video_writer.release()
    if args.show:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    elapsed = time.time() - start_time
    fps = processed_source_frames / max(elapsed, 1e-6)

    print('[DONE] video inference finished')
    print(f'[DONE] input_video            = {args.video}')
    print(f'[DONE] out_video              = {args.out_video}')
    print(f'[DONE] out_jsonl              = {args.out_jsonl}')
    print(f'[DONE] out_defects_json       = {out_defects_json}')
    print(f'[DONE] out_frame_index_json   = {out_frame_index_json}')
    print(f'[DONE] defect_images_dir      = {save_defect_frames_dir}')
    if save_defect_crops_dir:
        print(f'[DONE] defect_crops_dir       = {save_defect_crops_dir}')
    print(f'[DONE] class_map              = {class_map_path}')
    print(f'[DONE] source_frames          = {processed_source_frames}')
    print(f'[DONE] dedup_defects          = {len(exported_defects)}')
    print(f'[DONE] dedup_images           = {len(frame_index_records) * 2}')
    print(f'[DONE] avg_fps                = {fps:.2f}')
    print(f'[DONE] infer_stride           = {args.infer_stride}')
    print(f'[DONE] vis_stride             = {args.vis_stride}')
    print(f'[DONE] write_stride           = {args.write_stride}')
    print(f'[DONE] resize                 = {args.resize_width}x{args.resize_height}')
    print(f'[DONE] match_iou_thr          = {args.match_iou_thr}')
    print(f'[DONE] match_center_ratio     = {args.match_center_dist_ratio}')
    print(f'[DONE] max_frame_gap          = {args.max_frame_gap}')


if __name__ == '__main__':
    main()

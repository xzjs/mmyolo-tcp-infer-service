#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""TCP inference server for piping detection.

协议：
1. 客户端发送一行 JSON；
2. 服务端连续返回 JSON 行：
   - {"type":"ack", ...}
   - {"type":"progress", ...}
   - {"type":"result", "status":"ok", ...}
   - {"type":"error", "status":"error", ...}

本版相对仓库原版新增：
1. 真正把 infer_stride / vis_stride / write_stride 透传给视频脚本；
2. 支持缺陷级去重参数透传；
3. 支持返回 defects.json、frame_index.json、缺陷图片目录；
4. 文件内保留中文注释，便于交付和后续维护。
"""

import argparse
import inspect
import json
import os
import re
import socketserver
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple

import mmcv
import torch
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

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
PIPING_SRC = REPO_ROOT / 'projects' / 'piping' / 'src'

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PIPING_SRC) not in sys.path:
    sys.path.insert(0, str(PIPING_SRC))

from mmyolo.registry import VISUALIZERS
from mmyolo.utils.misc import get_file_list

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.ts'}
WINDOWS_PATH_RE = re.compile(r'^[a-zA-Z]:[\\/]')


def parse_path_mappings(raw_value: str) -> List[Tuple[str, str]]:
    value = (raw_value or '').strip()
    if not value:
        return []

    mappings: List[Tuple[str, str]] = []
    if value.startswith('{'):
        loaded = json.loads(value)
        if not isinstance(loaded, dict):
            raise ValueError('PATH_MAPPINGS JSON must be an object')
        for host_path, container_path in loaded.items():
            mappings.append((str(host_path).strip(), str(container_path).strip()))
        return mappings

    for item in value.split(';'):
        part = item.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError(f'invalid mapping `{part}`, expected HOST=CONTAINER')
        host_path, container_path = part.split('=', 1)
        host_path = host_path.strip()
        container_path = container_path.strip()
        if not host_path or not container_path:
            raise ValueError(f'invalid mapping `{part}`, host/container cannot be empty')
        mappings.append((host_path, container_path))
    return mappings


def get_pred_level(data_sample: Any) -> List[Tuple[int, int]]:
    if hasattr(data_sample, 'metainfo') and isinstance(data_sample.metainfo, dict):
        return data_sample.metainfo.get('pred_level', []) or []
    if hasattr(data_sample, '_metainfo') and isinstance(data_sample._metainfo, dict):
        return data_sample._metainfo.get('pred_level', []) or []
    return []


def parse_detection_result(
    data_sample: Any,
    score_thr: float,
    class_names: List[str],
    topk: Optional[int] = None,
) -> List[Dict[str, Any]]:
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
    if topk is not None and topk > 0:
        detections = detections[:topk]
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


def normalize_jsonable(data: Any) -> Any:
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, dict):
        return {str(k): normalize_jsonable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [normalize_jsonable(v) for v in data]
    return str(data)


class InferenceService:
    def __init__(
        self,
        default_config: str,
        default_checkpoint: str,
        default_device: str,
        default_score_thr: float,
        default_out_dir: str,
        default_infer_stride: int,
        default_vis_stride: int,
        default_write_stride: int,
        default_resize_width: int,
        default_resize_height: int,
        default_vis_score_thr: float,
        default_vis_topk: int,
        default_match_iou_thr: float,
        default_match_center_dist_ratio: float,
        default_max_frame_gap: int,
        default_post_merge_gap_frames: int,
        default_post_merge_iou_thr: float,
        default_chain_gap_frames: int,
        default_chain_iou_thr: float,
        default_chain_center_dist_ratio: float,
        default_chain_max_keep: int,
        default_min_hits: int,
        default_single_hit_high_score: float,
        default_max_frames: int,
        default_progress_every: int,
        path_mappings: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.default_config = default_config
        self.default_checkpoint = default_checkpoint
        self.default_device = default_device
        self.default_score_thr = default_score_thr
        self.default_out_dir = default_out_dir
        self.default_infer_stride = default_infer_stride
        self.default_vis_stride = default_vis_stride
        self.default_write_stride = default_write_stride
        self.default_resize_width = default_resize_width
        self.default_resize_height = default_resize_height
        self.default_vis_score_thr = default_vis_score_thr
        self.default_vis_topk = default_vis_topk
        self.default_match_iou_thr = default_match_iou_thr
        self.default_match_center_dist_ratio = default_match_center_dist_ratio
        self.default_max_frame_gap = default_max_frame_gap
        self.default_post_merge_gap_frames = default_post_merge_gap_frames
        self.default_post_merge_iou_thr = default_post_merge_iou_thr
        self.default_chain_gap_frames = default_chain_gap_frames
        self.default_chain_iou_thr = default_chain_iou_thr
        self.default_chain_center_dist_ratio = default_chain_center_dist_ratio
        self.default_chain_max_keep = default_chain_max_keep
        self.default_min_hits = default_min_hits
        self.default_single_hit_high_score = default_single_hit_high_score
        self.default_max_frames = default_max_frames
        self.default_progress_every = default_progress_every

        self._model_cache: Dict[Tuple[str, str, str], Any] = {}
        self._model_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._path_mappings = self._normalize_path_mappings(path_mappings or [])

    def _resolve_local_path(self, path_str: str, base_dir: Optional[Path] = None) -> Path:
        path = Path(path_str).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        return path.resolve()

    def _normalize_windows_path(self, path_str: str) -> str:
        path = path_str.strip().replace('/', '\\')
        path = re.sub(r'\\+', r'\\', path)
        if len(path) >= 2 and path[1] == ':':
            path = path[0].upper() + path[1:]
        return path.rstrip('\\')

    def _normalize_path_mappings(self, mappings: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        normalized: List[Tuple[str, str]] = []
        for host_path, container_path in mappings:
            host_norm = self._normalize_windows_path(host_path)
            container_norm = str(PurePosixPath(container_path))
            normalized.append((host_norm, container_norm))
        normalized.sort(key=lambda x: len(x[0]), reverse=True)
        return normalized

    def _is_windows_style_path(self, path_str: str) -> bool:
        return WINDOWS_PATH_RE.match(path_str.strip()) is not None

    def _map_windows_path_to_container(self, path_str: str) -> Optional[Path]:
        client_path = self._normalize_windows_path(path_str)
        client_lower = client_path.lower()

        for host_prefix, container_prefix in self._path_mappings:
            host_lower = host_prefix.lower()
            if client_lower == host_lower or client_lower.startswith(host_lower + '\\'):
                rel = client_path[len(host_prefix):].lstrip('\\')
                mapped = PurePosixPath(container_prefix)
                if rel:
                    mapped = mapped / PurePosixPath(rel.replace('\\', '/'))
                return Path(str(mapped)).resolve()

        drive = client_path[0].lower()
        rel_rest = client_path[2:].lstrip('\\').replace('\\', '/')
        candidates = [
            Path(f'/host_mnt/{drive}/{rel_rest}'),
            Path(f'/mnt/{drive}/{rel_rest}'),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _resolve_request_path(self, path_str: str, base_dir: Optional[Path], field_name: str, must_exist: bool) -> Path:
        raw = str(path_str).strip()
        if not raw:
            raise ValueError(f'`{field_name}` is empty')

        if self._is_windows_style_path(raw):
            mapped = self._map_windows_path_to_container(raw)
            if mapped is None:
                raise FileNotFoundError(
                    f'{field_name} is windows path `{raw}` but cannot map to container path. '
                    f'Please configure PATH_MAPPINGS and mount the host directory into container.'
                )
            path = mapped
        else:
            path = self._resolve_local_path(raw, base_dir=base_dir)

        if must_exist and not path.exists():
            raise FileNotFoundError(f'{field_name} path not found: {path}')
        return path

    def _get_model(self, config_path: Path, checkpoint_path: Path, device: str,
                   progress_cb: Callable[[int, str, str], None]) -> Any:
        cache_key = (str(config_path), str(checkpoint_path), device)
        with self._model_lock:
            if cache_key in self._model_cache:
                progress_cb(5, 'load_model', 'using cached model')
                return self._model_cache[cache_key]

            progress_cb(5, 'load_model', 'initializing detector')
            model = init_detector(str(config_path), str(checkpoint_path), device=device)
            self._model_cache[cache_key] = model
            return model

    def _infer(self, model: Any, file_path: str) -> Any:
        with self._infer_lock:
            return inference_detector(model, file_path)

    def _detect_kind(self, input_path: Path) -> str:
        if input_path.is_dir():
            return 'image'
        suffix = input_path.suffix.lower()
        if suffix in VIDEO_EXTS:
            return 'video'
        if suffix in IMAGE_EXTS:
            return 'image'
        raise ValueError(f'unsupported input type: {input_path}')

    def _image_out_name(self, input_path: Path, file_path: str) -> str:
        path_obj = Path(file_path)
        if input_path.is_dir():
            try:
                rel = path_obj.resolve().relative_to(input_path.resolve())
                return str(rel).replace(os.sep, '_')
            except Exception:
                return path_obj.name
        return path_obj.name

    def _process_images(
        self,
        task_id: str,
        input_path: Path,
        model: Any,
        score_thr: float,
        out_dir: Path,
        save_visualization: bool,
        progress_cb: Callable[[int, str, str], None],
    ) -> Dict[str, Any]:
        if input_path.is_dir():
            files, _ = get_file_list(str(input_path))
        else:
            files = [str(input_path)]

        if not files:
            raise ValueError(f'No image files found from {input_path}')

        mkdir_or_exist(str(out_dir))
        output_jsonl = out_dir / f'{task_id}_images.jsonl'

        class_names = list(getattr(model, 'dataset_meta', {}).get('classes', []))
        visualizer = None
        if save_visualization:
            visualizer = VISUALIZERS.build(model.cfg.visualizer)
            visualizer.dataset_meta = model.dataset_meta

        records: List[Dict[str, Any]] = []
        total = len(files)
        with open(output_jsonl, 'w', encoding='utf-8') as f_jsonl:
            for idx, file in enumerate(files, start=1):
                progress = min(10 + int((idx - 1) * 80 / max(total, 1)), 90)
                progress_cb(progress, 'infer_image', f'processing image {idx}/{total}')

                result = self._infer(model, file)
                detections = parse_detection_result(result, score_thr, class_names)
                grading = parse_grading_result(get_pred_level(result), class_names)
                record = {'image': file, 'detections': detections, 'pred_level': grading}
                records.append(record)
                f_jsonl.write(json.dumps(record, ensure_ascii=False) + '\n')

                if save_visualization and visualizer is not None:
                    img = mmcv.imread(file)
                    if img is not None:
                        img = mmcv.imconvert(img, 'bgr', 'rgb')
                        out_name = self._image_out_name(input_path, file)
                        out_file = out_dir / f'vis_{out_name}'
                        visualizer.add_datasample(
                            out_name,
                            img,
                            data_sample=result,
                            draw_gt=False,
                            show=False,
                            out_file=str(out_file),
                            pred_score_thr=score_thr,
                        )

        progress_cb(95, 'finalizing', 'image inference finished, preparing response')
        return {
            'kind': 'image',
            'input': str(input_path),
            'total_images': total,
            'output_jsonl': str(output_jsonl),
            'records': records,
        }

    def _process_video(
        self,
        task_id: str,
        input_path: Path,
        config_path: Path,
        checkpoint_path: Path,
        device: str,
        score_thr: float,
        out_dir: Path,
        infer_stride: int,
        vis_stride: int,
        write_stride: int,
        resize_width: int,
        resize_height: int,
        vis_score_thr: float,
        vis_topk: int,
        match_iou_thr: float,
        match_center_dist_ratio: float,
        max_frame_gap: int,
        post_merge_gap_frames: int,
        post_merge_iou_thr: float,
        chain_gap_frames: int,
        chain_iou_thr: float,
        chain_center_dist_ratio: float,
        chain_max_keep: int,
        min_hits: int,
        single_hit_high_score: float,
        max_frames: int,
        progress_every_frames: int,
        return_records: bool,
        progress_cb: Callable[[int, str, str], None],
    ) -> Dict[str, Any]:
        mkdir_or_exist(str(out_dir))
        output_jsonl = out_dir / f'{task_id}_video.jsonl'
        output_video = out_dir / f'{task_id}_video.mp4'
        output_defects_json = out_dir / f'{task_id}_video_defects.json'
        output_frame_index_json = out_dir / f'{task_id}_video_frame_index.json'
        output_defect_frames_dir = out_dir / f'{task_id}_video_defect_images'
        output_defect_crops_dir = out_dir / f'{task_id}_video_defect_crops_unused'

        script_path = REPO_ROOT / 'tools' / 'piping_infer' / 'video_infer_piping.py'
        if not script_path.exists():
            raise FileNotFoundError(f'video infer script not found: {script_path}')

        progress_cb(8, 'video_prepare', 'using video_infer_piping.py logic')

        total_frames = -1
        expected_total = -1
        try:
            reader = mmcv.VideoReader(str(input_path))
            if len(reader) > 0:
                total_frames = int(len(reader))
                expected_total = min(total_frames, max_frames) if max_frames > 0 else total_frames
        except Exception:
            pass

        cmd = [
            sys.executable,
            str(script_path),
            str(input_path),
            str(config_path),
            str(checkpoint_path),
            '--device', device,
            '--score-thr', str(score_thr),
            '--out-video', str(output_video),
            '--out-jsonl', str(output_jsonl),
            '--out-defects-json', str(output_defects_json),
            '--out-frame-index-json', str(output_frame_index_json),
            '--save-defect-frames-dir', str(output_defect_frames_dir),
            '--save-defect-crops-dir', str(output_defect_crops_dir),
            '--infer-stride', str(max(infer_stride, 1)),
            '--vis-stride', str(max(vis_stride, 1)),
            '--write-stride', str(max(write_stride, 1)),
            '--resize-width', str(max(resize_width, 32)),
            '--resize-height', str(max(resize_height, 32)),
            '--vis-score-thr', str(vis_score_thr),
            '--vis-topk', str(max(vis_topk, 1)),
            '--match-iou-thr', str(match_iou_thr),
            '--match-center-dist-ratio', str(match_center_dist_ratio),
            '--max-frame-gap', str(max(max_frame_gap, 1)),
            '--post-merge-gap-frames', str(max(post_merge_gap_frames, 0)),
            '--post-merge-iou-thr', str(post_merge_iou_thr),
            '--chain-gap-frames', str(max(chain_gap_frames, 0)),
            '--chain-iou-thr', str(chain_iou_thr),
            '--chain-center-dist-ratio', str(chain_center_dist_ratio),
            '--chain-max-keep', str(max(chain_max_keep, 1)),
            '--min-hits', str(max(min_hits, 1)),
            '--single-hit-high-score', str(single_hit_high_score),
            '--print-every', str(max(progress_every_frames, 1)),
        ]
        if max_frames > 0:
            cmd.extend(['--max-frames', str(max_frames)])

        env = os.environ.copy()
        py_path_items = [str(REPO_ROOT), str(PIPING_SRC)]
        old_py_path = env.get('PYTHONPATH', '').strip()
        if old_py_path:
            py_path_items.append(old_py_path)
        env['PYTHONPATH'] = os.pathsep.join(py_path_items)

        progress_cb(10, 'video_script_start', 'starting video_infer_piping.py')
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            bufsize=1,
        )

        processed_frames = -1
        inferenced_frames = -1
        dedup_defects = -1
        tail_logs: List[str] = []
        if proc.stdout is None:
            raise RuntimeError('failed to read video script stdout')

        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            tail_logs.append(line)
            if len(tail_logs) > 100:
                tail_logs = tail_logs[-100:]

            info_match = re.search(r'processed=(\d+)', line)
            if info_match:
                processed_frames = int(info_match.group(1))
                if expected_total > 0:
                    progress = 10 + int(80 * processed_frames / max(expected_total, 1))
                    progress = min(progress, 90)
                else:
                    progress = 50
                progress_cb(progress, 'infer_video', line)
                continue

            done_frames_match = re.search(r'^\[DONE\]\s+source_frames\s*=\s*(\d+)', line)
            if done_frames_match:
                processed_frames = int(done_frames_match.group(1))
                progress_cb(95, 'video_done', line)
                continue

            dedup_match = re.search(r'^\[DONE\]\s+dedup_defects\s*=\s*(\d+)', line)
            if dedup_match:
                dedup_defects = int(dedup_match.group(1))
                progress_cb(96, 'video_dedup', line)
                continue

            progress_cb(15, 'video_log', line)

        return_code = proc.wait()
        if return_code != 0:
            joined = '\n'.join(tail_logs[-25:])
            raise RuntimeError(f'video_infer_piping.py failed with code {return_code}\nlast logs:\n{joined}')

        records: List[Dict[str, Any]] = []
        if output_jsonl.exists():
            with open(output_jsonl, 'r', encoding='utf-8') as f_jsonl:
                for line in f_jsonl:
                    line = line.strip()
                    if not line:
                        continue
                    if return_records:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            try:
                with open(output_jsonl, 'r', encoding='utf-8') as f_jsonl:
                    inferenced_frames = sum(1 for _ in f_jsonl)
            except Exception:
                inferenced_frames = -1

        defect_records: List[Dict[str, Any]] = []
        if output_defects_json.exists():
            try:
                with open(output_defects_json, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    defect_records = loaded.get('items', []) if isinstance(loaded, dict) else []
                    if dedup_defects < 0:
                        dedup_defects = len(defect_records)
            except Exception:
                defect_records = []

        progress_cb(98, 'finalizing', 'video inference finished, preparing response')
        return {
            'kind': 'video',
            'input': str(input_path),
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'inferenced_frames': inferenced_frames,
            'dedup_defects': dedup_defects,
            'requested_infer_stride': max(infer_stride, 1),
            'requested_vis_stride': max(vis_stride, 1),
            'requested_write_stride': max(write_stride, 1),
            'output_video': str(output_video),
            'output_jsonl': str(output_jsonl),
            'output_defects_json': str(output_defects_json),
            'output_frame_index_json': str(output_frame_index_json),
            'output_defect_images_dir': str(output_defect_frames_dir),
            'output_defect_frames_dir': str(output_defect_frames_dir),
            'output_defect_crops_dir': str(output_defect_crops_dir),
            'records': records,
            'defects': defect_records if return_records else [],
            'engine': 'video_infer_piping.py',
        }

    def process_request(self, request: Dict[str, Any], progress_cb: Callable[[int, str, str], None]) -> Dict[str, Any]:
        file_path = str(request.get('file_path', '')).strip()
        if not file_path:
            raise ValueError('`file_path` is required')

        task_id = str(request.get('task_id', '')).strip() or uuid.uuid4().hex
        base_dir = Path(request.get('base_dir', REPO_ROOT))
        input_path = self._resolve_request_path(file_path, base_dir=base_dir, field_name='file_path', must_exist=True)

        config_value = str(request.get('config', '')).strip() or self.default_config
        checkpoint_value = str(request.get('checkpoint', '')).strip() or self.default_checkpoint
        device = str(request.get('device', '')).strip() or self.default_device
        score_thr = float(request.get('score_thr', self.default_score_thr))
        out_dir_value = str(request.get('out_dir', '')).strip() or self.default_out_dir

        infer_stride = int(request.get('infer_stride', self.default_infer_stride))
        vis_stride = int(request.get('vis_stride', self.default_vis_stride))
        write_stride = int(request.get('write_stride', self.default_write_stride))
        resize_width = int(request.get('resize_width', self.default_resize_width))
        resize_height = int(request.get('resize_height', self.default_resize_height))
        vis_score_thr = float(request.get('vis_score_thr', self.default_vis_score_thr))
        vis_topk = int(request.get('vis_topk', self.default_vis_topk))

        match_iou_thr = float(request.get('match_iou_thr', self.default_match_iou_thr))
        match_center_dist_ratio = float(request.get('match_center_dist_ratio', self.default_match_center_dist_ratio))
        max_frame_gap = int(request.get('max_frame_gap', self.default_max_frame_gap))
        post_merge_gap_frames = int(request.get('post_merge_gap_frames', self.default_post_merge_gap_frames))
        post_merge_iou_thr = float(request.get('post_merge_iou_thr', self.default_post_merge_iou_thr))
        chain_gap_frames = int(request.get('chain_gap_frames', self.default_chain_gap_frames))
        chain_iou_thr = float(request.get('chain_iou_thr', self.default_chain_iou_thr))
        chain_center_dist_ratio = float(request.get('chain_center_dist_ratio', self.default_chain_center_dist_ratio))
        chain_max_keep = int(request.get('chain_max_keep', self.default_chain_max_keep))
        min_hits = int(request.get('min_hits', self.default_min_hits))
        single_hit_high_score = float(request.get('single_hit_high_score', self.default_single_hit_high_score))

        max_frames = int(request.get('max_frames', self.default_max_frames))
        progress_every_frames = int(request.get('progress_every_frames', self.default_progress_every))
        save_visualization = bool(request.get('save_visualization', False))
        return_records = bool(request.get('return_records', False))

        config_path = self._resolve_request_path(config_value, base_dir=REPO_ROOT, field_name='config', must_exist=True)
        checkpoint_path = self._resolve_request_path(checkpoint_value, base_dir=REPO_ROOT, field_name='checkpoint', must_exist=True)
        out_dir = self._resolve_request_path(out_dir_value, base_dir=REPO_ROOT, field_name='out_dir', must_exist=False)
        mkdir_or_exist(str(out_dir))

        progress_cb(2, 'request_received', f'task_id={task_id}')
        kind = self._detect_kind(input_path)
        progress_cb(3, 'input_ready', f'input kind: {kind}')

        if kind == 'video':
            result = self._process_video(
                task_id=task_id,
                input_path=input_path,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                device=device,
                score_thr=score_thr,
                out_dir=out_dir,
                infer_stride=infer_stride,
                vis_stride=vis_stride,
                write_stride=write_stride,
                resize_width=resize_width,
                resize_height=resize_height,
                vis_score_thr=vis_score_thr,
                vis_topk=vis_topk,
                match_iou_thr=match_iou_thr,
                match_center_dist_ratio=match_center_dist_ratio,
                max_frame_gap=max_frame_gap,
                post_merge_gap_frames=post_merge_gap_frames,
                post_merge_iou_thr=post_merge_iou_thr,
                chain_gap_frames=chain_gap_frames,
                chain_iou_thr=chain_iou_thr,
                chain_center_dist_ratio=chain_center_dist_ratio,
                chain_max_keep=chain_max_keep,
                min_hits=min_hits,
                single_hit_high_score=single_hit_high_score,
                max_frames=max_frames,
                progress_every_frames=progress_every_frames,
                return_records=return_records,
                progress_cb=progress_cb,
            )
        else:
            model = self._get_model(config_path, checkpoint_path, device, progress_cb)
            result = self._process_images(
                task_id=task_id,
                input_path=input_path,
                model=model,
                score_thr=score_thr,
                out_dir=out_dir,
                save_visualization=save_visualization,
                progress_cb=progress_cb,
            )

        progress_cb(100, 'done', 'task completed')
        return {
            'task_id': task_id,
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'request_file_path': file_path,
            'resolved_input_path': str(input_path),
            'config': str(config_path),
            'checkpoint': str(checkpoint_path),
            'device': device,
            'score_thr': score_thr,
            'result': result,
        }


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class InferenceRequestHandler(socketserver.StreamRequestHandler):
    def _send_json(self, payload: Dict[str, Any]) -> None:
        data = json.dumps(normalize_jsonable(payload), ensure_ascii=False) + '\n'
        self.wfile.write(data.encode('utf-8'))
        self.wfile.flush()

    def handle(self) -> None:
        service: InferenceService = self.server.inference_service  # type: ignore[attr-defined]
        line = self.rfile.readline()
        if not line:
            return
        try:
            request = json.loads(line.decode('utf-8').strip())
            if not isinstance(request, dict):
                raise ValueError('request body must be a JSON object')
        except Exception as exc:
            self._send_json({'type': 'error', 'status': 'error', 'message': f'invalid request json: {exc}'})
            return

        task_id = str(request.get('task_id', '')).strip() or uuid.uuid4().hex
        request['task_id'] = task_id
        self._send_json({'type': 'ack', 'status': 'accepted', 'task_id': task_id})

        def progress_cb(progress: int, stage: str, message: str) -> None:
            self._send_json({
                'type': 'progress',
                'status': 'running',
                'task_id': task_id,
                'progress': max(0, min(int(progress), 100)),
                'stage': stage,
                'message': message,
            })

        try:
            result = service.process_request(request, progress_cb)
            self._send_json({'type': 'result', 'status': 'ok', 'task_id': task_id, 'result': result})
        except Exception as exc:
            self._send_json({
                'type': 'error',
                'status': 'error',
                'task_id': task_id,
                'message': str(exc),
                'traceback': traceback.format_exc(),
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='TCP server for piping inference')
    parser.add_argument('--host', default=os.getenv('TCP_HOST', '0.0.0.0'), help='TCP bind host')
    parser.add_argument('--port', type=int, default=int(os.getenv('TCP_PORT', '9000')), help='TCP bind port')
    parser.add_argument('--default-config', default=os.getenv('MODEL_CONFIG', str(REPO_ROOT / 'projects/piping/configs/yolov8_s_fast_8xb16-500e_ours.py')), help='default config path if not provided in request')
    parser.add_argument('--default-checkpoint', default=os.getenv('MODEL_CHECKPOINT', str(REPO_ROOT / 'weights/best_coco_破裂_precision_epoch_25.pth')), help='default checkpoint path if not provided in request')
    parser.add_argument('--default-device', default=os.getenv('MODEL_DEVICE', 'cuda:0'), help='default inference device if not provided in request')
    parser.add_argument('--default-score-thr', type=float, default=float(os.getenv('SCORE_THR', '0.3')), help='default score threshold if not provided in request')
    parser.add_argument('--default-out-dir', default=os.getenv('OUTPUT_DIR', str(REPO_ROOT / 'output/tcp_results')), help='default output directory for jsonl and visualization')

    parser.add_argument('--default-infer-stride', type=int, default=int(os.getenv('INFER_STRIDE', '3')), help='default infer stride for video')
    parser.add_argument('--default-vis-stride', type=int, default=int(os.getenv('VIS_STRIDE', '3')), help='default vis stride for video')
    parser.add_argument('--default-write-stride', type=int, default=int(os.getenv('WRITE_STRIDE', '3')), help='default write stride for video')
    parser.add_argument('--default-resize-width', type=int, default=int(os.getenv('RESIZE_WIDTH', '1280')), help='default resize width for video')
    parser.add_argument('--default-resize-height', type=int, default=int(os.getenv('RESIZE_HEIGHT', '720')), help='default resize height for video')
    parser.add_argument('--default-vis-score-thr', type=float, default=float(os.getenv('VIS_SCORE_THR', '0.18')), help='default visualization score threshold for video')
    parser.add_argument('--default-vis-topk', type=int, default=int(os.getenv('VIS_TOPK', '10')), help='default visualization topk for video')

    parser.add_argument('--default-match-iou-thr', type=float, default=float(os.getenv('MATCH_IOU_THR', '0.35')), help='default cross-frame match IoU threshold')
    parser.add_argument('--default-match-center-dist-ratio', type=float, default=float(os.getenv('MATCH_CENTER_DIST_RATIO', '0.22')), help='default normalized center distance threshold')
    parser.add_argument('--default-max-frame-gap', type=int, default=int(os.getenv('MAX_FRAME_GAP', '9')), help='default max frame gap for a track')
    parser.add_argument('--default-post-merge-gap-frames', type=int, default=int(os.getenv('POST_MERGE_GAP_FRAMES', '9')), help='default second-pass merge frame gap')
    parser.add_argument('--default-post-merge-iou-thr', type=float, default=float(os.getenv('POST_MERGE_IOU_THR', '0.30')), help='default second-pass merge IoU threshold')
    parser.add_argument('--default-chain-gap-frames', type=int, default=int(os.getenv('CHAIN_GAP_FRAMES', '18')), help='default defect-chain merge frame gap')
    parser.add_argument('--default-chain-iou-thr', type=float, default=float(os.getenv('CHAIN_IOU_THR', '0.18')), help='default defect-chain merge IoU threshold')
    parser.add_argument('--default-chain-center-dist-ratio', type=float, default=float(os.getenv('CHAIN_CENTER_DIST_RATIO', '0.28')), help='default defect-chain normalized center distance threshold')
    parser.add_argument('--default-chain-max-keep', type=int, default=int(os.getenv('CHAIN_MAX_KEEP', '1')), help='default max kept samples for one repeated defect chain')
    parser.add_argument('--default-min-hits', type=int, default=int(os.getenv('MIN_HITS', '2')), help='default min hits to keep a defect')
    parser.add_argument('--default-single-hit-high-score', type=float, default=float(os.getenv('SINGLE_HIT_HIGH_SCORE', '0.45')), help='default score threshold to keep a single-hit defect')

    parser.add_argument('--default-max-frames', type=int, default=int(os.getenv('MAX_FRAMES', '-1')), help='default max frames for video, -1 means all')
    parser.add_argument('--default-progress-every', type=int, default=int(os.getenv('PROGRESS_EVERY_FRAMES', '30')), help='default progress interval for video frames')
    parser.add_argument('--path-mappings', default=os.getenv('PATH_MAPPINGS', ''), help='windows->container path mappings: "C:\\\\host\\\\data=/data/input;D:\\\\videos=/data/videos" or JSON object string')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_mappings = parse_path_mappings(args.path_mappings)

    inference_service = InferenceService(
        default_config=args.default_config,
        default_checkpoint=args.default_checkpoint,
        default_device=args.default_device,
        default_score_thr=args.default_score_thr,
        default_out_dir=args.default_out_dir,
        default_infer_stride=args.default_infer_stride,
        default_vis_stride=args.default_vis_stride,
        default_write_stride=args.default_write_stride,
        default_resize_width=args.default_resize_width,
        default_resize_height=args.default_resize_height,
        default_vis_score_thr=args.default_vis_score_thr,
        default_vis_topk=args.default_vis_topk,
        default_match_iou_thr=args.default_match_iou_thr,
        default_match_center_dist_ratio=args.default_match_center_dist_ratio,
        default_max_frame_gap=args.default_max_frame_gap,
        default_post_merge_gap_frames=args.default_post_merge_gap_frames,
        default_post_merge_iou_thr=args.default_post_merge_iou_thr,
        default_chain_gap_frames=args.default_chain_gap_frames,
        default_chain_iou_thr=args.default_chain_iou_thr,
        default_chain_center_dist_ratio=args.default_chain_center_dist_ratio,
        default_chain_max_keep=args.default_chain_max_keep,
        default_min_hits=args.default_min_hits,
        default_single_hit_high_score=args.default_single_hit_high_score,
        default_max_frames=args.default_max_frames,
        default_progress_every=args.default_progress_every,
        path_mappings=path_mappings,
    )

    with ThreadedTCPServer((args.host, args.port), InferenceRequestHandler) as server:
        server.inference_service = inference_service  # type: ignore[attr-defined]
        print(f'[TCP] server listening on {args.host}:{args.port}')
        print(f'[TCP] default config     : {args.default_config}')
        print(f'[TCP] default checkpoint : {args.default_checkpoint}')
        print(f'[TCP] default device     : {args.default_device}')
        print(f'[TCP] default out_dir    : {args.default_out_dir}')
        print(f'[TCP] default infer/vis/write stride : {args.default_infer_stride}/{args.default_vis_stride}/{args.default_write_stride}')
        print(f'[TCP] default resize     : {args.default_resize_width}x{args.default_resize_height}')
        print(f'[TCP] default dedup      : iou={args.default_match_iou_thr}, center_ratio={args.default_match_center_dist_ratio}, max_gap={args.default_max_frame_gap}, min_hits={args.default_min_hits}')
        print(f'[TCP] default chain merge: gap={args.default_chain_gap_frames}, iou={args.default_chain_iou_thr}, center_ratio={args.default_chain_center_dist_ratio}, max_keep={args.default_chain_max_keep}')
        if path_mappings:
            print('[TCP] path mappings:')
            for host_path, container_path in path_mappings:
                print(f'  - {host_path} -> {container_path}')
        else:
            print('[TCP] path mappings     : (empty)')
        server.serve_forever()


if __name__ == '__main__':
    main()

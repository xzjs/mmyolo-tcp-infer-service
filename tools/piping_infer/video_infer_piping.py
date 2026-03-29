# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse
# import json
# import os
# import sys
# import time
# from pathlib import Path
# from typing import Any, Dict, List, Tuple

# import cv2
# import mmcv
# from mmcv.transforms import Compose
# from mmdet.apis import inference_detector, init_detector
# from mmengine.utils import mkdir_or_exist



# # 每多少帧推理一次
# INFER_STRIDE = 2

# # 每多少帧做一次可视化
# VIS_STRIDE = 2

# # 每多少帧写一次输出视频
# WRITE_STRIDE = 2

# # 输出与推理使用的分辨率
# RESIZE_WIDTH = 640
# RESIZE_HEIGHT = 360

# # =========================================================
# # 控制画面不要太乱
# # =========================================================

# # 只保留高分框
# VIS_SCORE_THR = 0.15

# # 每帧最多画几个框
# VIS_TOPK = 14

# # 是否画类别文字
# DRAW_LABEL_TEXT = True

# # 文字字体大小
# TEXT_SCALE = 0.6

# # 文字粗细
# TEXT_THICKNESS = 2

# # 线框粗细
# BOX_THICKNESS = 2


# # 先把项目根目录和自定义模块目录加入 PYTHONPATH。
# THIS_FILE = Path(__file__).resolve()
# REPO_ROOT = THIS_FILE.parents[2]
# PIPING_SRC = REPO_ROOT / 'projects' / 'piping' / 'src'

# if str(REPO_ROOT) not in sys.path:
#     sys.path.insert(0, str(REPO_ROOT))
# if str(PIPING_SRC) not in sys.path:
#     sys.path.insert(0, str(PIPING_SRC))


# def parse_args() -> argparse.Namespace:
#     """解析命令行参数。"""
#     parser = argparse.ArgumentParser(description='Piping video inference script')
#     parser.add_argument('video', help='输入视频路径')
#     parser.add_argument('config', help='模型配置文件路径')
#     parser.add_argument('checkpoint', help='模型权重路径')
#     parser.add_argument('--device', default='cuda:0', help='推理设备，例如 cuda:0 或 cpu')
#     parser.add_argument('--score-thr', type=float, default=0.3, help='检测框置信度阈值（保留参数，但实际可视化阈值用 VIS_SCORE_THR）')
#     parser.add_argument('--out-video', required=True, help='输出可视化视频路径')
#     parser.add_argument('--out-jsonl', required=True, help='输出逐帧 JSONL 路径')
#     parser.add_argument('--show', action='store_true', help='是否实时弹窗显示结果')
#     parser.add_argument('--wait-time', type=int, default=1, help='显示窗口等待时间，单位毫秒')
#     parser.add_argument('--max-frames', type=int, default=-1, help='最多处理多少帧，-1 表示全部处理')
#     parser.add_argument('--save-vis-frames-dir', default='', help='可选，额外保存逐帧可视化图片目录')
#     parser.add_argument('--print-every', type=int, default=30, help='每隔多少帧打印一次进度')
#     return parser.parse_args()


# def ensure_parent(path: str) -> None:
#     """确保输出文件的父目录存在。"""
#     parent = os.path.dirname(os.path.abspath(path))
#     if parent:
#         mkdir_or_exist(parent)


# def get_pred_level(data_sample: Any) -> List[Tuple[int, int]]:
#     """从 DetDataSample 中安全读取 pred_level 元信息。"""
#     if hasattr(data_sample, 'metainfo') and isinstance(data_sample.metainfo, dict):
#         return data_sample.metainfo.get('pred_level', []) or []
#     if hasattr(data_sample, '_metainfo') and isinstance(data_sample._metainfo, dict):
#         return data_sample._metainfo.get('pred_level', []) or []
#     try:
#         meta = data_sample.metainfo_keys()
#         if 'pred_level' in meta:
#             return data_sample.get('pred_level', []) or []
#     except Exception:
#         pass
#     return []


# def to_python_number(x: Any) -> Any:
#     """把 numpy / tensor 标量转成 Python 原生数值。"""
#     if hasattr(x, 'item'):
#         try:
#             return x.item()
#         except Exception:
#             return x
#     return x


# def parse_detection_result(
#     data_sample: Any,
#     score_thr: float,
#     class_names: List[str],
#     topk: int,
# ) -> List[Dict[str, Any]]:
#     """解析检测结果，只保留高分前 topk 个框。"""
#     detections: List[Dict[str, Any]] = []

#     if not hasattr(data_sample, 'pred_instances'):
#         return detections

#     pred_instances = data_sample.pred_instances
#     if not hasattr(pred_instances, 'scores'):
#         return detections

#     scores = pred_instances.scores.detach().cpu().numpy()
#     labels = pred_instances.labels.detach().cpu().numpy()
#     bboxes = pred_instances.bboxes.detach().cpu().numpy()

#     for score, label, bbox in zip(scores, labels, bboxes):
#         score_f = float(score)
#         if score_f < score_thr:
#             continue

#         label_id = int(label)
#         label_name = class_names[label_id] if 0 <= label_id < len(class_names) else str(label_id)

#         detections.append({
#             'label_id': label_id,
#             'label_name': label_name,
#             'score': round(score_f, 6),
#             'bbox_xyxy': [round(float(v), 2) for v in bbox.tolist()],
#         })

#     detections.sort(key=lambda x: x['score'], reverse=True)
#     detections = detections[:topk]
#     return detections


# def parse_grading_result(pred_level: List[Tuple[int, int]], class_names: List[str]) -> List[Dict[str, Any]]:
#     """把 pred_level 解析成更清晰的 JSON 结构。"""
#     grading: List[Dict[str, Any]] = []
#     for item in pred_level:
#         if not isinstance(item, (list, tuple)) or len(item) != 2:
#             continue
#         cls_id = int(to_python_number(item[0]))
#         grade = int(to_python_number(item[1]))
#         cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
#         grading.append({
#             'class_id': cls_id,
#             'class_name': cls_name,
#             'grade': grade,
#         })
#     return grading


# def draw_grade_summary(frame: Any, grading: List[Dict[str, Any]]) -> Any:
#     """在视频左上角叠加等级摘要。"""
#     if not grading:
#         return frame

#     overlay_lines = ['frame-level grades:']
#     for item in grading:
#         overlay_lines.append(f"cls_{item['class_id']} -> grade_{item['grade']}")

#     y = 28
#     for line in overlay_lines:
#         cv2.putText(
#             frame,
#             line,
#             (12, y),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 255, 255),
#             2,
#             cv2.LINE_AA,
#         )
#         y += 28
#     return frame


# def draw_simple_bboxes(frame: Any, detections: List[Dict[str, Any]]) -> Any:
#     """只画 bbox，默认不画类别文字。"""
#     h, w = frame.shape[:2]

#     for det in detections:
#         x1, y1, x2, y2 = [int(v) for v in det['bbox_xyxy']]

#         x1 = max(0, min(x1, w - 1))
#         y1 = max(0, min(y1, h - 1))
#         x2 = max(0, min(x2, w - 1))
#         y2 = max(0, min(y2, h - 1))

#         if x2 <= x1 or y2 <= y1:
#             continue

#         cv2.rectangle(
#             frame,
#             (x1, y1),
#             (x2, y2),
#             (0, 255, 0),
#             BOX_THICKNESS,
#         )

#         if DRAW_LABEL_TEXT:
#             text = f"{det['label_name']} {det['score']:.2f}"
#             cv2.putText(
#                 frame,
#                 text,
#                 (x1, max(y1 - 5, 0)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 TEXT_SCALE,
#                 (0, 255, 0),
#                 TEXT_THICKNESS,
#                 cv2.LINE_AA,
#             )

#     return frame


# def main() -> None:
#     """主函数。"""
#     args = parse_args()

#     if not os.path.isfile(args.video):
#         raise FileNotFoundError(f'输入视频不存在: {args.video}')
#     if not os.path.isfile(args.config):
#         raise FileNotFoundError(f'配置文件不存在: {args.config}')
#     if not os.path.isfile(args.checkpoint):
#         raise FileNotFoundError(f'模型权重不存在: {args.checkpoint}')

#     ensure_parent(args.out_video)
#     ensure_parent(args.out_jsonl)
#     if args.save_vis_frames_dir:
#         mkdir_or_exist(args.save_vis_frames_dir)

#     # 构建模型
#     model = init_detector(args.config, args.checkpoint, device=args.device)

#     # 视频帧输入改为 ndarray
#     model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
#     test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

#     # 类别名
#     if hasattr(model, 'dataset_meta') and isinstance(model.dataset_meta, dict):
#         class_names = list(model.dataset_meta.get('classes', []))
#     else:
#         class_names = []

#     # 打开视频
#     video_reader = mmcv.VideoReader(args.video)
#     if len(video_reader) == 0:
#         raise RuntimeError(f'无法读取视频内容: {args.video}')

#     src_fps = float(video_reader.fps)

#     # 因为不是每帧都写，所以输出 fps 也对应降低
#     out_fps = max(src_fps / WRITE_STRIDE, 1.0)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_writer = cv2.VideoWriter(
#         args.out_video,
#         fourcc,
#         out_fps,
#         (RESIZE_WIDTH, RESIZE_HEIGHT),
#     )
#     if not video_writer.isOpened():
#         raise RuntimeError(f'输出视频无法写入: {args.out_video}')

#     # 写 classes 映射
#     class_map_path = os.path.splitext(args.out_jsonl)[0] + '_classes.json'
#     with open(class_map_path, 'w', encoding='utf-8') as f:
#         json.dump(
#             {
#                 'classes': class_names,
#                 'note': '如果使用的是 grading / level 模型，则 pred_level 为帧级按类别聚合后的等级结果。'
#             },
#             f,
#             ensure_ascii=False,
#             indent=2,
#         )

#     start_time = time.time()
#     processed = 0
#     frame_idx = -1

#     # 缓存最近一次推理结果，仅用于当前抽样帧的后续处理
#     last_result = None
#     last_vis_frame_bgr = None
#     last_detections: List[Dict[str, Any]] = []
#     last_grading: List[Dict[str, Any]] = []

#     with open(args.out_jsonl, 'w', encoding='utf-8') as f_jsonl:
#         for frame in video_reader:
#             frame_idx += 1
#             if frame is None:
#                 continue

#             if args.max_frames > 0 and processed >= args.max_frames:
#                 break

#             # 统一先缩放到较小分辨率
#             frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

#             need_infer = (frame_idx % INFER_STRIDE == 0)
#             need_vis = (frame_idx % VIS_STRIDE == 0)
#             need_write = (frame_idx % WRITE_STRIDE == 0)

#             # 1) 不再每一帧都推理
#             if need_infer:
#                 result = inference_detector(model, frame, test_pipeline=test_pipeline)
#                 detections = parse_detection_result(
#                     result,
#                     VIS_SCORE_THR,
#                     class_names,
#                     VIS_TOPK,
#                 )
#                 pred_level = get_pred_level(result)
#                 grading = parse_grading_result(pred_level, class_names)

#                 last_result = result
#                 last_detections = detections
#                 last_grading = grading

#                 # 只记录实际做了推理的帧
#                 record = {
#                     'frame_index': frame_idx,
#                     'time_sec': round(frame_idx / src_fps, 6),
#                     'detections': detections,
#                     'pred_level': grading,
#                 }
#                 f_jsonl.write(json.dumps(record, ensure_ascii=False) + '\n')

#             # 2) 不再每一帧都做完整可视化
#             if need_vis and last_result is not None:
#                 vis_frame = frame.copy()
#                 vis_frame = draw_simple_bboxes(vis_frame, last_detections)
#                 vis_frame = draw_grade_summary(vis_frame, last_grading)
#                 last_vis_frame_bgr = vis_frame

#                 if args.save_vis_frames_dir:
#                     out_img = os.path.join(args.save_vis_frames_dir, f'frame_{frame_idx:06d}.jpg')
#                     mmcv.imwrite(last_vis_frame_bgr, out_img)

#                 if args.show:
#                     cv2.namedWindow('piping_video_infer', 0)
#                     mmcv.imshow(last_vis_frame_bgr, 'piping_video_infer', args.wait_time)

#             # 3) 不再每一帧都写 mp4
#             if need_write and last_vis_frame_bgr is not None:
#                 video_writer.write(last_vis_frame_bgr)

#             processed += 1
#             if processed % max(args.print_every, 1) == 0:
#                 elapsed = time.time() - start_time
#                 fps = processed / max(elapsed, 1e-6)
#                 print(f'[INFO] processed={processed}, frame_idx={frame_idx}, avg_fps={fps:.2f}')

#     video_writer.release()
#     if args.show:
#         try:
#             cv2.destroyAllWindows()
#         except cv2.error:
#             pass

#     elapsed = time.time() - start_time
#     fps = processed / max(elapsed, 1e-6)
#     print('[DONE] video inference finished')
#     print(f'[DONE] input_video   = {args.video}')
#     print(f'[DONE] out_video     = {args.out_video}')
#     print(f'[DONE] out_jsonl     = {args.out_jsonl}')
#     print(f'[DONE] class_map     = {class_map_path}')
#     print(f'[DONE] frames        = {processed}')
#     print(f'[DONE] avg_fps       = {fps:.2f}')
#     print(f'[DONE] infer_stride  = {INFER_STRIDE}')
#     print(f'[DONE] vis_stride    = {VIS_STRIDE}')
#     print(f'[DONE] write_stride  = {WRITE_STRIDE}')
#     print(f'[DONE] resize        = {RESIZE_WIDTH}x{RESIZE_HEIGHT}')


# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
管道缺陷视频推理脚本（中文标签可视化版）

特点：
1. 不再每一帧都推理；
2. 不再每一帧都做完整可视化；
3. 不再每一帧都写 mp4；
4. 不再保持原视频分辨率；
5. 框上中文类别名使用 PIL 绘制，避免 OpenCV 中文乱码/不显示。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import mmcv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mmcv.transforms import Compose
from mmdet.apis import inference_detector, init_detector
from mmengine.utils import mkdir_or_exist


# =========================================================
# 只改这几类推理/可视化参数
# =========================================================

# # 每多少帧做一次推理
# INFER_STRIDE = 5

# # 每多少帧做一次可视化
# VIS_STRIDE = 5

# # 每多少帧写一次输出视频
# WRITE_STRIDE = 5

# # 输出与推理使用的分辨率
# RESIZE_WIDTH = 640
# RESIZE_HEIGHT = 360

# # 可视化阈值：只画高于该阈值的框
# VIS_SCORE_THR = 0.15

# # 每帧最多显示前多少个框
# VIS_TOPK = 14

# # 是否在框上画类别名和分数
# DRAW_LABEL_TEXT = True

INFER_STRIDE = 3
VIS_STRIDE = 3
WRITE_STRIDE = 3
RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 720
VIS_SCORE_THR = 0.18
VIS_TOPK = 10
DRAW_LABEL_TEXT = True

# 框线宽
BOX_THICKNESS = 2

# 文字缩放（PIL 方案里主要看 FONT_SIZE，这里只保留兼容）
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2

# 中文字体路径：请尽量用系统中真实存在的中文字体
# AutoDL 常见可用：
# /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc
# /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
FONT_PATH = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
FONT_SIZE = 20

# 左上角等级摘要是否画中文类名（这里仍默认画 cls_id，避免太挤）
DRAW_GRADE_SUMMARY = True


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


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='Piping video inference script')
    parser.add_argument('video', help='输入视频路径')
    parser.add_argument('config', help='模型配置文件路径')
    parser.add_argument('checkpoint', help='模型权重路径')
    parser.add_argument('--device', default='cuda:0', help='推理设备，例如 cuda:0 或 cpu')
    parser.add_argument('--score-thr', type=float, default=0.3, help='检测框置信度阈值（保留接口）')
    parser.add_argument('--out-video', required=True, help='输出可视化视频路径')
    parser.add_argument('--out-jsonl', required=True, help='输出逐帧 JSONL 路径')
    parser.add_argument('--show', action='store_true', help='是否实时弹窗显示结果')
    parser.add_argument('--wait-time', type=int, default=1, help='显示窗口等待时间，单位毫秒')
    parser.add_argument('--max-frames', type=int, default=-1, help='最多处理多少帧，-1 表示全部处理')
    parser.add_argument('--save-vis-frames-dir', default='', help='可选，额外保存逐帧可视化图片目录')
    parser.add_argument('--print-every', type=int, default=30, help='每隔多少帧打印一次进度')
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    """确保输出文件的父目录存在。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        mkdir_or_exist(parent)


def get_pred_level(data_sample: Any) -> List[Tuple[int, int]]:
    """从 DetDataSample 中安全读取 pred_level 元信息。"""
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
    """把 numpy / tensor 标量转成 Python 原生数值。"""
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
    detections = detections[:topk]
    return detections


def parse_grading_result(pred_level: List[Tuple[int, int]], class_names: List[str]) -> List[Dict[str, Any]]:
    """把 pred_level 解析成更清晰的 JSON 结构。"""
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


def get_pil_font(size: int) -> ImageFont.FreeTypeFont:
    """尽量加载中文字体，失败则退回默认字体。"""
    candidate_fonts = [
        FONT_PATH,
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for font_path in candidate_fonts:
        try:
            if os.path.isfile(font_path):
                return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text_pil(
    frame_bgr: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_size: int = FONT_SIZE,
    text_color=(0, 0, 0),
    bg_color=(0, 255, 0),
) -> np.ndarray:
    """用 PIL 在 BGR 图像上绘制中文文本。"""

    def _safe_text(raw_text: str) -> str:
        """默认字体不支持中文时，降级为可编码文本，避免流程崩溃。"""
        try:
            raw_text.encode('latin-1')
            return raw_text
        except UnicodeEncodeError:
            return raw_text.encode('latin-1', errors='replace').decode('latin-1')

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = get_pil_font(font_size)
    text_to_draw = text

    # 兼容不同 Pillow 版本：优先 textbbox，不支持则退回 textsize/getsize。
    try:
        x1, y1, x2, y2 = draw.textbbox((x, y), text_to_draw, font=font)
    except Exception:
        text_to_draw = _safe_text(text_to_draw)
        try:
            text_w, text_h = draw.textsize(text_to_draw, font=font)
        except Exception:
            text_w, text_h = font.getsize(text_to_draw)
        x1, y1 = x, y
        x2, y2 = x + text_w, y + text_h

    pad = 3
    draw.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad], fill=bg_color)
    try:
        draw.text((x, y), text_to_draw, font=font, fill=text_color)
    except Exception:
        draw.text((x, y), _safe_text(text_to_draw), font=font, fill=text_color)

    out_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return out_bgr


def draw_grade_summary(frame: Any, grading: List[Dict[str, Any]]) -> Any:
    """在视频左上角叠加等级摘要。"""
    if not DRAW_GRADE_SUMMARY:
        return frame
    if not grading:
        return frame

    overlay_lines = ['frame-level grades:']
    for item in grading:
        overlay_lines.append(f"cls_{item['class_id']} -> grade_{item['grade']}")

    y = 28
    for line in overlay_lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28
    return frame


def draw_simple_bboxes(frame: Any, detections: List[Dict[str, Any]]) -> Any:
    """画 bbox，并支持中文类别名显示。"""
    h, w = frame.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox_xyxy']]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            BOX_THICKNESS,
        )

        if DRAW_LABEL_TEXT:
            text = f"{det['label_name']} {det['score']:.2f}"
            text_y = max(y1 - FONT_SIZE - 6, 0)
            frame = draw_text_pil(frame, text, x1, text_y)

    return frame


def main() -> None:
    """主函数。"""
    args = parse_args()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(f'输入视频不存在: {args.video}')
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f'配置文件不存在: {args.config}')
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f'模型权重不存在: {args.checkpoint}')

    ensure_parent(args.out_video)
    ensure_parent(args.out_jsonl)
    if args.save_vis_frames_dir:
        mkdir_or_exist(args.save_vis_frames_dir)

    # 构建模型
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # 视频帧输入改为 ndarray
    model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # 类别名
    if hasattr(model, 'dataset_meta') and isinstance(model.dataset_meta, dict):
        class_names = list(model.dataset_meta.get('classes', []))
    else:
        class_names = []

    # 打开视频
    video_reader = mmcv.VideoReader(args.video)
    if len(video_reader) == 0:
        raise RuntimeError(f'无法读取视频内容: {args.video}')

    src_fps = float(video_reader.fps)

    # 输出视频写入器
    out_fps = max(src_fps / WRITE_STRIDE, 1.0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        args.out_video,
        fourcc,
        out_fps,
        (RESIZE_WIDTH, RESIZE_HEIGHT),
    )
    if not video_writer.isOpened():
        raise RuntimeError(f'输出视频无法写入: {args.out_video}')

    # 写 classes 映射
    class_map_path = os.path.splitext(args.out_jsonl)[0] + '_classes.json'
    with open(class_map_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'classes': class_names,
                'note': '如果使用的是 grading / level 模型，则 pred_level 为帧级按类别聚合后的等级结果。'
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    start_time = time.time()
    processed = 0
    frame_idx = -1

    # 缓存最近一次推理结果，仅用于当前抽样帧的后续处理
    last_result = None
    last_vis_frame_bgr = None
    last_detections: List[Dict[str, Any]] = []
    last_grading: List[Dict[str, Any]] = []

    window_created = False

    with open(args.out_jsonl, 'w', encoding='utf-8') as f_jsonl:
        for frame in video_reader:
            frame_idx += 1
            if frame is None:
                continue

            if args.max_frames > 0 and processed >= args.max_frames:
                break

            # 统一先缩放到较小分辨率
            frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

            need_infer = (frame_idx % INFER_STRIDE == 0)
            need_vis = (frame_idx % VIS_STRIDE == 0)
            need_write = (frame_idx % WRITE_STRIDE == 0)

            # 1) 不再每一帧都推理
            if need_infer:
                result = inference_detector(model, frame, test_pipeline=test_pipeline)
                detections = parse_detection_result(
                    result,
                    VIS_SCORE_THR,
                    class_names,
                    VIS_TOPK,
                )
                pred_level = get_pred_level(result)
                grading = parse_grading_result(pred_level, class_names)

                last_result = result
                last_detections = detections
                last_grading = grading

                # 只记录实际做了推理的帧
                record = {
                    'frame_index': frame_idx,
                    'time_sec': round(frame_idx / src_fps, 6),
                    'detections': detections,
                    'pred_level': grading,
                }
                f_jsonl.write(json.dumps(record, ensure_ascii=False) + '\n')

            # 2) 不再每一帧都做完整可视化
            if need_vis and last_result is not None:
                vis_frame = frame.copy()
                vis_frame = draw_simple_bboxes(vis_frame, last_detections)
                vis_frame = draw_grade_summary(vis_frame, last_grading)
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

            # 3) 不再每一帧都写 mp4
            if need_write and last_vis_frame_bgr is not None:
                video_writer.write(last_vis_frame_bgr)

            processed += 1
            if processed % max(args.print_every, 1) == 0:
                elapsed = time.time() - start_time
                fps = processed / max(elapsed, 1e-6)
                print(f'[INFO] processed={processed}, frame_idx={frame_idx}, avg_fps={fps:.2f}')

    video_writer.release()

    if args.show:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    elapsed = time.time() - start_time
    fps = processed / max(elapsed, 1e-6)
    print('[DONE] video inference finished')
    print(f'[DONE] input_video   = {args.video}')
    print(f'[DONE] out_video     = {args.out_video}')
    print(f'[DONE] out_jsonl     = {args.out_jsonl}')
    print(f'[DONE] class_map     = {class_map_path}')
    print(f'[DONE] frames        = {processed}')
    print(f'[DONE] avg_fps       = {fps:.2f}')
    print(f'[DONE] infer_stride  = {INFER_STRIDE}')
    print(f'[DONE] vis_stride    = {VIS_STRIDE}')
    print(f'[DONE] write_stride  = {WRITE_STRIDE}')
    print(f'[DONE] resize        = {RESIZE_WIDTH}x{RESIZE_HEIGHT}')


if __name__ == '__main__':
    main()

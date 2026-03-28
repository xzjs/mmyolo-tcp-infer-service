#!/usr/bin/env python3
"""
filter_normal_bg_plus_targets.py
生成新的 COCO 标注文件：
  - 所有含目标的图片（含标注）全部保留；
  - 位于 normal_images 目录的纯背景图片（无标注）也保留；
  - 其余纯背景图片丢弃；
  - 图片目录不改动。
"""

import json, argparse
from pathlib import Path
from typing import Dict, Any, Set, List

def is_under_normal(file_name: str) -> bool:
    """判断图片路径是否在 normal_images 目录下"""
    return "normal_images" in Path(file_name).parts

def main(args):
    coco = json.load(open(args.in_json, encoding='utf-8'))

    # 1. 找出所有含目标的图片 id
    target_img_ids: Set[int] = {ann['image_id'] for ann in coco['annotations']}

    # 2. 找出所有纯背景图片
    bg_imgs = [img for img in coco['images'] if img['id'] not in target_img_ids]

    # 3. 仅保留位于 normal_images 的纯背景图片
    bg_imgs_keep = [img for img in bg_imgs if is_under_normal(img['file_name'])]
    bg_img_ids_keep: Set[int] = {img['id'] for img in bg_imgs_keep}

    # 4. 最终要保留的图片 id
    keep_img_ids = target_img_ids | bg_img_ids_keep

    # 5. 过滤 images
    new_images = [img for img in coco['images'] if img['id'] in keep_img_ids]

    # 6. 过滤 annotations（只保留保留图片的标注）
    new_anns = [ann for ann in coco['annotations'] if ann['image_id'] in keep_img_ids]

    # 7. 构建新 JSON
    new_coco = {
        "images": new_images,
        "annotations": new_anns,
        "categories": coco.get("categories", [])
    }

    # 8. 写出
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(new_coco, open(out_path, 'w', encoding='utf-8'), indent=2)

    print(f"已生成新标注文件：{out_path}")
    print(f"保留图片 {len(new_images)} 张（含目标 {len(target_img_ids)} 张 + "
          f"normal_images 纯背景 {len(bg_imgs_keep)} 张）")
    print(f"保留标注 {len(new_anns)} 条")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="生成含目标图片 + normal_images 纯背景图片的新 COCO 标注文件"
    )
    parser.add_argument("in_json",  help="原始 COCO 标注文件")
    parser.add_argument("out_json", help="输出过滤后的 COCO 标注文件")
    args = parser.parse_args()
    main(args)
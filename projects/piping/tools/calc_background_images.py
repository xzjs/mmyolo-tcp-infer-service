from __future__ import annotations
import os
import json



def calc_background_ratio(coco_dict: dict) -> tuple[int, float, int]:
    """
    统计 COCO 数据集中纯背景图片的比例
    :param coco_dict: 已加载的 COCO dict
    :return: (纯背景图片数, 纯背景占比, 总图片数)
    """
    img_ids_with_ann = {ann['image_id'] for ann in coco_dict['annotations']}
    total_imgs = len(coco_dict['images'])
    bg_imgs = total_imgs - len(img_ids_with_ann)
    ratio = bg_imgs / total_imgs if total_imgs else 0.0
    normal_imgs = 0
    for img in coco_dict['images']:
        if img['file_name'].startswith('normal_images'):
            normal_imgs += 1
    return bg_imgs, ratio, total_imgs, normal_imgs

input_dir = '/data/ours_annotation/annotations/annotations_v4/'
files = os.listdir(input_dir)

for f in files:
    if f.startswith('train') and 'coco' in f and f.endswith('.json'):
        with open(os.path.join(input_dir,f)) as fo:
            coco = json.load(fo)
        bg_imgs, ratio, total_imgs, normal_imgs = calc_background_ratio(coco)
        print(f"{f}: Total images: {total_imgs}, background images: {bg_imgs}, ratio {ratio}, normal images:{normal_imgs}")
  
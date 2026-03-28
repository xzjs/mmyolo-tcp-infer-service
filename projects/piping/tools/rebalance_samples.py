import json
import random
from collections import defaultdict
from pycocotools.coco import COCO
from tqdm import tqdm
import argparse
import copy
import numpy as np

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def oversample_balance(ann_path, out_path, seed=42):
    random.seed(seed)
    data = load_json(ann_path)
    coco = COCO(ann_path)

    # 1. 统计每个 (cat, level) 的缺陷数量
    cat_level_counts = defaultdict(int)
    img_to_cat_level = defaultdict(lambda: defaultdict(int))
    for ann in data['annotations']:
        key = (ann['category_id'], ann['bbox_level'])
        cat_level_counts[key] += 1
        img_to_cat_level[ann['image_id']][key] += 1

    # 2. 计算目标数量 = 所有组合中最大者
    target = max(list(cat_level_counts.values()))
    print(f"Target defects per (category, level): {target}")

    # 3. 计算每个 (cat, level) 还需要多少缺陷
    need = {k: target - v for k, v in cat_level_counts.items()}

    # 4. 计算每张图片可提供的 (cat, level) 计数
    img_candidates = list(img_to_cat_level.keys())
    img_contrib = {
        img_id: img_to_cat_level[img_id]
        for img_id in img_candidates
    }

    # 5. 贪心+随机地挑选图片，直到所有 need <= 0
    selected_img_ids = []  # 允许重复
    pbar = tqdm(total=sum(need.values()), desc='Sampling images')
    while any(v > 0 for v in need.values()):
        # 随机打乱顺序，避免偏向列表前部
        random.shuffle(img_candidates)
        added = False
        for img_id in img_candidates:
            contrib = img_contrib[img_id]
            # 只要这张图片至少还能贡献一个“缺口”就选用
            if any(need[k] > 0 and contrib.get(k, 0) > 0 for k in need):
                selected_img_ids.append(img_id)
                for k, cnt in contrib.items():
                    need[k] -= cnt
                pbar.update(sum(contrib[k] for k in need if need[k] > 0))
                added = True
                break
        if not added:
            raise RuntimeError("无法继续采样，请检查数据分布或降低目标")
    pbar.close()

    # 6. 构建新的 images / annotations
    new_images = []
    new_anns = []
    img_id_map = {}  # old_img_id -> list of new_img_ids
    max_img_id = max(data['images'], key=lambda x: x['id'])['id'] + 1
    max_ann_id = max(data['annotations'], key=lambda x: x['id'])['id'] + 1

    # 先加入原始图片
    
    for img_info in data['images']:
        img_id_map[img_info['id']] = [img_info['id']]
    new_images.extend(data['images'])

    # 再处理重复采样的图片
    for old_img_id in selected_img_ids:
        old_img = coco.imgs[old_img_id]
        new_img = copy.deepcopy(old_img)
        new_img['id'] = max_img_id
        max_img_id += 1
        new_images.append(new_img)
        img_id_map.setdefault(old_img_id, []).append(new_img['id'])

        # 复制该图片的所有标注
        for ann in coco.imgToAnns[old_img_id]:
            new_ann = copy.deepcopy(ann)
            new_ann['id'] = max_ann_id
            max_ann_id += 1
            new_ann['image_id'] = new_img['id']
            new_anns.append(new_ann)

    # 7. 合并原始标注与新增标注
    new_anns.extend(data['annotations'])

    # 8. 生成最终 json
    new_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': data['categories'],
        'images': new_images,
        'annotations': new_anns
    }

    save_json(new_data, out_path)
    print(f"Oversampled annotations saved to {out_path}")

    # 9. 验证
    final_counts = defaultdict(int)
    for ann in new_anns:
        final_counts[(ann['category_id'], ann['bbox_level'])] += 1
    print("Final counts per (category, level):")
    for k in sorted(final_counts.keys()):
        print(f"  {k}: {final_counts[k]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json', help='原始 COCO 标注文件')
    parser.add_argument('output_json', help='平衡后保存路径')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    oversample_balance(args.input_json, args.output_json, seed=args.seed)
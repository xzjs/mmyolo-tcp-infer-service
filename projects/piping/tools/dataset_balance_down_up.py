"""
balance_coco.py
处理带 bbox_level 的 COCO 标注：
1. 降采样指定类别/等级
2. 过采样其余类别
3. 绘制三阶段分布图
"""

import json, cv2, os, math, random
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, Set, Tuple, List
import copy
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']   # 黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示为方块的问题

IMG_ROOT='/data/ours_annotation/images'
# 需要降采样的 (category_name, level)
DOWN_TARGETS = {
    ("支管暗接", 1),
    ("树根", 1),
    ("破裂", 1),
    ("破裂", 2),
    ("错口", 1),
    ("错口", 2),
}

FILTER_TARGETS={
    "起伏",
    "浮渣",
    "渗漏",
    "沉积",
    "结垢",
    "腐蚀",
    "脱节"
    
}

# ------------------ 工具函数 ------------------
def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, p):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def image_blur_score(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var() if img is not None else 0

def split_seq_frame(name: str):
    """
    返回 (seq_id, frame)
    例：cctv_001_20240605_000123.jpg -> ('cctv_001_20240605_', 123)
    """
    stem = Path(name).stem               # 去掉后缀
    # 找到最后一个连续数字
    import re
    m = re.search(r'(\d+)(?=\D*$)', stem)
    if not m:
        return stem, 0                   # 没有数字，兜底
    frame_str = m.group(1)
    seq_id = stem[:m.start()]            # 数字之前的部分
    return seq_id, int(frame_str)


def plot_dataset_distribution(coco: Dict[str, Any],
                              out_png: str,
                              title: str = None):
    """
    绘制缺陷数量条形图，并在最右侧增加“纯背景”柱
    """
    # 1. 缺陷统计 (category, level)
    # counts = defaultdict(int)
    class_counts = defaultdict(int)
    cat_id2name = {c['id']: c['name'] for c in coco['categories']}
    for ann in coco['annotations']:
        #key = (cat_id2name[ann['category_id']], ann['bbox_level'])
        #ounts[key] += 1
        class_counts[cat_id2name[ann['category_id']]] += 1
    print(class_counts)

    # 2. 纯背景图片数
    img_ids_with_ann = {ann['image_id'] for ann in coco['annotations']}
    bg_count = len(coco['images']) - len(img_ids_with_ann)

    # 3. 构造 DataFrame：缺陷 + 背景
    rows = [(cat, lvl, cnt) for (cat, lvl), cnt in counts.items()]
    if bg_count > 0:                       # 只有存在背景时才添加
        rows.append(('纯背景', 0, bg_count))   # Level 用 0 占位
    df = pd.DataFrame(rows, columns=['Category', 'Level', 'Count'])

    # 4. 画图
    sns.set_theme(style="whitegrid", font='SimHei', font_scale=1.4)
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.4), 5))

    # 为“纯背景”单独指定颜色
    palette = sns.color_palette("tab10", n_colors=len(counts))
    palette += [(0.5, 0.5, 0.5)]  # 灰色给背景

    sns.barplot(
        x='Category',
        y='Count',
        hue='Level',
        data=df,
        palette=palette,
        dodge=False,        # 背景无分组，避免堆叠
        ax=ax
    )

    if title is None:
        title = 'Defect & Background Count Distribution'
    ax.set_title(title, fontsize=14)

    # 5. 信息文字
    num_imgs = len(coco['images'])
    num_anns = len(coco['annotations'])
    ax.text(0.02, 0.95,
            f"总样本：{num_imgs} 张图 / {num_anns} 个框 / 背景 {bg_count} 张",
            transform=ax.transAxes,
            verticalalignment='bottom',
            fontproperties='SimHei',
            fontsize=20,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f'Distribution plot saved to {out_png}')
'''
def plot_dataset_distribution(coco: Dict[str, Any],
                              out_png: str,
                              title: str = None):
    """
    缺陷：每类一组，每组内按 level 分柱；
    背景：最右侧单独一组，仅一根柱。
    """
    # 1. 缺陷统计
    counts = defaultdict(int)
    cat_id2name = {c['id']: c['name'] for c in coco['categories']}
    for ann in coco['annotations']:
        key = (cat_id2name[ann['category_id']], ann['bbox_level'])
        counts[key] += 1

    # 2. 纯背景数量
    img_ids_with_ann = {ann['image_id'] for ann in coco['annotations']}
    bg_count = len(coco['images']) - len(img_ids_with_ann)

    # 3. 构造 DataFrame
    rows = [(cat, lvl, cnt) for (cat, lvl), cnt in counts.items()]
    if bg_count > 0:
        rows.append(('纯背景', 0, bg_count))   # Level=0 仅作占位
    df = pd.DataFrame(rows, columns=['Category', 'Level', 'Count'])
    # 让“纯背景”排在最后
    df['Category'] = pd.Categorical(df['Category'],
                                    categories=list({c for c, _ in counts.keys()}) + ['纯背景'],
                                    ordered=True)
    df = df.sort_values(['Category', 'Level'])

    # 4. 画图
    sns.set_theme(style="whitegrid", font='SimHei', font_scale=1.4)
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.4), 5))

    sns.barplot(
        x='Category',
        y='Count',
        hue='Level',
        data=df,
        palette='tab10',
        dodge=True,
        ax=ax
    )

    # 5. 图例：把 Level=0 的“纯背景”显示为 “纯背景”
    handles, labels = ax.get_legend_handles_labels()
    labels = ['纯背景' if lab == '0' else f'Level {lab}' for lab in labels]
    ax.legend(handles, labels, title='')

    if title is None:
        title = '缺陷 & 背景数量分布'
    ax.set_title(title, fontsize=14)

    # 6. 信息文字
    num_imgs = len(coco['images'])
    num_anns = len(coco['annotations'])
    ax.text(0.02, 0.95,
            f"总样本：{num_imgs} 张图 / {num_anns} 个框 / 背景 {bg_count} 张",
            transform=ax.transAxes,
            verticalalignment='bottom',
            fontproperties='SimHei',
            fontsize=20,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f'Distribution plot saved to {out_png}')

'''
def calc_background_ratio(coco_dict: dict) -> Tuple[int, float, int]:
    """
    统计 COCO 数据集中纯背景图片的比例
    :param coco_dict: 已加载的 COCO dict
    :return: (纯背景图片数, 纯背景占比, 总图片数)
    """
    img_ids_with_ann = {ann['image_id'] for ann in coco_dict['annotations']}
    total_imgs = len(coco_dict['images'])
    bg_imgs = total_imgs - len(img_ids_with_ann)
    ratio = bg_imgs / total_imgs if total_imgs else 0.0
    return bg_imgs, ratio, total_imgs

def downsample(coco: Dict[str, Any],
               down_targets: Set[Tuple[str, int]]=DOWN_TARGETS,
               img_root:str=IMG_ROOT) -> Dict[str, Any]:
    imgs      = {i['id']: i for i in coco['images']}
    cat2name  = {c['id']: c['name'] for c in coco['categories']}
    name2id   = {v: k for k, v in cat2name.items()}

    img2anns = defaultdict(list)
    for ann in coco['annotations']:
        img2anns[ann['image_id']].append(ann)

    cand_img_ids = set()
    keep_img_ids = set()

    for (cat_name, level) in down_targets:
        cid = name2id[cat_name]
        cand_anns = [a for a in coco['annotations']
                     if a['category_id'] == cid and a.get('bbox_level') == level]
        cand_img_ids.add(a['image_id'] for a in cand_anns)

        seq2items = defaultdict(list)
        for ann in cand_anns:
            img_name = imgs[ann['image_id']]['file_name']
            seq_id, _ = split_seq_frame(img_name)
            area = ann['bbox'][2] * ann['bbox'][3]
            score = image_blur_score(os.path.join(img_root, img_name))
            seq2items[seq_id].append((ann, area, score, ann['image_id']))

        for seq_id, items in seq2items.items():
            best = max(items, key=lambda x: (x[1], x[2]))
            keep_img_ids.append(best[3])

    keep_img_ids = [k for k in imgs.keys() if (k in cand_img_ids and k in keep_img_ids) or (not k in cand_img_ids)]
    down_images = [imgs[i] for i in keep_img_ids]
    down_anns   = [a for a in coco['annotations'] if a['image_id'] in keep_img_ids]
    coco_down = {**coco, 'images': down_images, 'annotations': down_anns}
    return coco_down

def oversample(coco: Dict[str, Any],
               target:str=None) -> Dict[str, Any]:
    imgs      = {i['id']: i for i in coco['images']}
    img2anns  = defaultdict(list)
    for ann in coco['annotations']:
        img2anns[ann['image_id']].append(ann)

    single_defect_imgs = [(img_id, anns[0])
                          for img_id, anns in img2anns.items() if len(anns) == 1]

    cat_level = lambda ann: (next(c['name'] for c in coco['categories']
                                  if c['id'] == ann['category_id']),
                             ann.get('bbox_level', 1))

    cl2single = defaultdict(list)
    for img_id, ann in single_defect_imgs:
        cl2single[cat_level(ann)].append(img_id)

    counts = Counter(cat_level(a) for a in coco['annotations'])
    if target == 'max':
        target_cnt = max(counts.values())
    if target == 'median':
        target_cnt = int(np.median(list(counts.values())))

    print(f"Oversample taget number:{target_cnt}")
    cl2need = {cl: target_cnt - cnt for cl, cnt in counts.items()}

    final_images = coco['images'].copy()
    final_anns   = coco['annotations'].copy()

    next_img_id = max(imgs.keys()) + 1
    next_ann_id = max(a['id'] for a in final_anns) + 1

    for cl, need in cl2need.items():
        if need <= 0: continue
        pool = cl2single[cl]
        if not pool:
            print(f"警告：{cl} 无单缺陷图，跳过过采样")
            continue
        random.shuffle(pool)
        for i in range(need):
            img_id = pool[i % len(pool)]
            old_img = imgs[img_id]
            new_img = {**old_img, 'id': next_img_id}

            ann = img2anns[img_id][0]
            new_ann = {**ann, 'id': next_ann_id, 'image_id': next_img_id}

            final_images.append(new_img)
            final_anns.append(new_ann)
            next_img_id += 1
            next_ann_id += 1
        final_coco = {**coco, 'images': final_images, 'annotations': final_anns}
    return final_coco

def sample_background(coco: Dict[str, Any],
                      bg_ratio: float) -> Dict[str, Any]:
    imgs = {i['id']: i for i in coco['images']}
    img_ids_with_targets = {ann['image_id'] for ann in coco['annotations']}

    fg_imgs = [img for img in coco['images'] if img['id'] in img_ids_with_targets]

    bg_pool = [img for img in coco['images']
               if img['id'] not in img_ids_with_targets
               ]

    n_targets = len(img_ids_with_targets)
    n_bg_need = int(n_targets * bg_ratio / (1 - bg_ratio)) if bg_ratio < 1 else 0
    n_bg_have = len(bg_pool)

    print(f'Number of images containing targets: {n_targets}')
    print(f"Number of background images: {n_bg_have}")

    if n_bg_have == 0:
        print("不含纯背景图像，无需抽样。")
        return coco
    elif n_bg_need <= 0 :
        chosen = []
    elif n_bg_need <= n_bg_have:
        chosen = random.sample(bg_pool, n_bg_need)
    else:
        chosen = bg_pool * (n_bg_need // n_bg_have) + \
                 random.sample(bg_pool, n_bg_need % n_bg_have)
        random.shuffle(chosen)

    bg_imgs = []
    next_img_id = max(imgs.keys())+1
    for img in chosen:
        new_img = {**img, 'id': next_img_id}
        bg_imgs.append(new_img)
        next_img_id += 1
    print(f'Number of images containing targets after sampling: {len(fg_imgs)}')
    print(f"Number of background images after sampling: {len(bg_imgs)}")
    final_images = fg_imgs+bg_imgs
    final_coco = {**coco, 'images': final_images}
    return final_coco


def drop_imgs_with_target_categories(coco: Dict[str, Any],
                                     target_cats: List[str]) -> Dict[str, Any]:
    """
    删除所有包含指定类别目标的图片、这些图片上的全部标注，
    并同步删除 categories 中对应的类别条目。

    Parameters
    ----------
    coco : dict
        已读入内存的 COCO 格式字典
    target_cats : list[str]
        要剔除的类别名称列表（大小写敏感）

    Returns
    -------
    dict
        过滤后的新 COCO 字典
    """
    coco = copy.deepcopy(coco)

    # 1. 类别名 -> id
    cat_name2id = {c['name']: c['id'] for c in coco['categories']}
    target_ids: Set[int] = {cat_name2id[name] for name in target_cats if name in cat_name2id}
    if not target_ids:           # 没有要删的类别，直接返回
        return coco

    # 2. 找出需要删除的图片 id
    bad_img_ids = {
        ann['image_id']
        for ann in coco['annotations']
        if ann['category_id'] in target_ids
    }

    # 3. 过滤图片
    coco['images'] = [img for img in coco['images'] if img['id'] not in bad_img_ids]

    # 4. 过滤标注：同时剔除两类
    #    a) 位于被删图片上的所有标注
    #    b) 属于被删类别的所有标注（即使图片未被删，也一并删除）
    coco['annotations'] = [
        ann for ann in coco['annotations']
        if ann['image_id'] not in bad_img_ids and ann['category_id'] not in target_ids
    ]

    # 5. 过滤 categories
    coco['categories'] = [c for c in coco['categories'] if c['id'] not in target_ids]

    # 6. （可选）重新连续编号 category_id，保持从 1 开始
    # 如果后续训练代码依赖连续 id，可取消下面注释
    """
    new_id_map = {old: new for new, old in enumerate([c['id'] for c in coco['categories']], start=1)}
    for c in coco['categories']:
        c['id'] = new_id_map[c['id']]
    for ann in coco['annotations']:
        ann['category_id'] = new_id_map[ann['category_id']]
    """

    return coco


# ------------------ 主流程 ------------------
def main():
    src_json = '/data/ours_annotation/annotations/annotations_v4/train_coco_format.json'
    coco = load_json(src_json)
    plot_dataset_distribution(coco, 'jjdjdj.png') 
    exit(1)
    coco = drop_imgs_with_target_categories(coco, target_cats=FILTER_TARGETS)
    output_json = '/data/ours_annotation/annotations/annotations_v4/train_normal_coco_format_with_level_filter_no_annotation_9classes.json'
    save_json(coco, output_json)
    plot_dataset_distribution(coco, 'normal_filter_no_annotation_9classes_distribution.png')
 
    # coco = oversample(coco, target='median') 
 
    bg_ratio = [0,0.05,0.1,0.2]
    for ratio in bg_ratio:
        new_coco = sample_background(coco, ratio)
        output_json=Path(src_json).parent/Path(f"train_normal_coco_format_with_level_filter_no_annotation_9classes_no_balance_bg_{ratio}.json")
        plot_dataset_distribution(new_coco, f'normal_filter_no_annotation_orignal_distribution_9classes_no_balance_bg_{ratio}.png')
        save_json(new_coco, output_json)
 
    # 处理测试集
    input_json = '/data/ours_annotation/annotations/annotations_v4/test_coco_format_with_level.json'
    coco = load_json(input_json)
    coco = drop_imgs_with_target_categories(coco, target_cats=FILTER_TARGETS)
    output_json = '/data/ours_annotation/annotations/annotations_v4/test_coco_format_with_level_9classes.json'
    save_json(coco, output_json)
   



if __name__ == "__main__":
    main()

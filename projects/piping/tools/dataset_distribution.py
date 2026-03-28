import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

import pandas as pd


plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']   # 黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示为方块的问题

import seaborn as sns



def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_before_balance(ann_path, out_png, title=None):
    data = load_json(ann_path)

    print(data['categories'])

    # 1. 统计 (category, level) 缺陷数量
    counts = defaultdict(int)
    cat_id2name = {c['id']: c['name'] for c in data['categories']}
    for ann in data['annotations']:
        key = (cat_id2name[ann['category_id']], ann['bbox_level'])
        counts[key] += 1

    nmin = min(counts.items(), key=lambda x:x[1])
    nmax = max(counts.items(), key=lambda x:x[1])

    print(f'数量最多类别{nmax[0]}样本个数{nmax[1]}')
    print(f'数量最少类别{nmin[0]}样本个数{nmin[1]}')

    # 2. 构造 DataFrame
    df = pd.DataFrame(
        [(cat, lvl, cnt) for (cat, lvl), cnt in counts.items()],
        columns=['Category', 'Level', 'Count']
    ).sort_values(['Category', 'Level'])

    # 3. 画图
    sns.set_theme(style="whitegrid",font='SimHei',font_scale=1.4)

    fig, ax = plt.subplots(figsize=(max(6, len(df)*0.4), 5))
    sns.barplot(
        x='Category',
        y='Count',
        hue='Level',
        data=df,
        palette='tab10'
    )
    if title is None:
        title = 'Defect Count Distribution (Before Balance)'
    plt.title(title, fontsize=14)
    num_imgs = len(data['images'])
    num_anns = len(data['annotations'])
    ax.text(0.02, 0.95,
        f"总样本：{num_imgs} 张图 / {num_anns} 个框",
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json', help='原始 COCO 标注文件')
    parser.add_argument('output_png', help='保存的 PNG 路径，如 before.png')
    parser.add_argument('--title', default=None, help='图标题')
    args = parser.parse_args()
    plot_before_balance(args.input_json, args.output_png, args.title)
import json
import argparse
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
import os.path as osp
import random

from typing import List, Dict


################################################
# 划分训练集和测试集：
# 1. 同一个管段的样本不能同时出现在训练集和测试集
# 2. 测试集中每个类别（含分级共58类，下同）的样本比较均衡
# 3. 测试集中每个类别中的标注的数量占总体的比例约为指定的比例
# 4. 测试集中每个类别中的缺陷（一个缺陷可能标注多次）数量占总体的比列约为指定的比例
################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json', type=str, required=True, help='COCO json label path')
    parser.add_argument(
        '--out-dir', type=str, required=True, help='output path')
    parser.add_argument(
        '--ratios',
        nargs='+',
        type=float,
        help='ratio for sub dataset, if set 2 number then will generate '
        'trainval + test (eg. "0.8 0.1 0.1" or "2 1 1"), if set 3 number '
        'then will generate train + val + test (eg. "0.85 0.15" or "2 1")')
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Whether to display in disorder')
    parser.add_argument('--seed', default=-1, type=int, help='seed')
    args = parser.parse_args()
    return args


def split_piping_dataset(json_path: str, save_dir: str, ratios: list,
                       shuffle: bool, seed: int):
    if not Path(json_path).exists():
        raise FileNotFoundError(f'Can not not found {json_path}')

    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)

    # ratio normalize
    ratios = np.array(ratios) / np.array(ratios).sum()
    

    assert len(ratios) == 2

    with open(json_path) as f:
        dataset = json.load(f)

    classnames = dataset["metainfo"]["classes"]

    img_anno_count_dict = {}
    piping_img_dict = defaultdict(list)
    piping_img_sequence_dict = defaultdict(set)
    img_sequence_count_dict = defaultdict(Counter)
    
    total_anno_count = Counter()

    for data in dataset['data_list']:
        if data["img_path"].endswith("gt.jpg"):
            continue
        img_path = data["img_path"]
        piping_id = osp.basename(img_path).split('_')[:2]
        piping_img_dict[' '.join(piping_id)].append(img_path) 
        img_sequence = osp.basename(img_path).rsplit('_', 1)[0]
        piping_img_sequence_dict[' '.join(piping_id)].add(img_sequence)

        defects = []
        for obj in data['objects']:
            defect_name = obj["defect_name"]
            defect_level = obj.get("defect_level", None)
            if defect_level is None:
                assert classnames[defect_name] == "连接"
                continue 
            defects.append(f"{classnames[defect_name]}-{defect_level}级")
        
        img_anno_count_dict[img_path] = Counter(defects)
        total_anno_count.update(defects)

        ori_counter = img_sequence_count_dict[img_sequence]
        cur_counter = img_anno_count_dict[img_path]
        max_counter = {key: max(ori_counter.get(key, 0), cur_counter.get(key, 0)) for key in set(ori_counter)| set(cur_counter)}
        img_sequence_count_dict[img_sequence] = Counter(max_counter)

    total_defect_count = Counter()
    for _, cnt in img_sequence_count_dict.items():
        total_defect_count.update(cnt)

    # pipings = list(piping_img_dict.keys())
    # print(len(pipings))
    # for i in range(10):
    #     print(pipings[i])
    #     print(piping_img_dict[pipings[i]])

    print(total_anno_count)
    for idx, item in enumerate(img_anno_count_dict.items()):
        print(item)
        if idx == 9:
            break

    print(f"Total Number of annotations: {sum(total_anno_count.values())}")

    print(total_defect_count)
    for idx, item in enumerate(img_sequence_count_dict.items()):
        print(item)
        if idx == 9:
            break

    print(f"Total Number of defects: {sum(total_defect_count.values())}")

    piping_ids = list(piping_img_dict.keys())
    print(f"Number of pipings :{len(piping_ids)}")
    print(f"Number of categories: {len(total_anno_count)}")
    num_pipings = len(piping_ids)
    test_size = int(num_pipings*ratios[1]*0.7)

    while True:
        test_pipings = random.sample(piping_ids, test_size)

        flag1 = check_ratio_of_annos(test_pipings, ratios[1], total_anno_count, piping_img_dict, img_anno_count_dict)
        if not flag1:
            continue
        flag2 = check_ratio_of_defects(test_pipings, ratios[1], total_defect_count, piping_img_sequence_dict, img_sequence_count_dict)

        if flag1 and flag2:
            break 

    test_dataset = {'metainfo':{'classes': classnames}, 'data_list':[]}
    train_dataset = {'metainfo':{'classes': classnames}, 'data_list':[]}

    print(test_pipings)

    for data in dataset['data_list']:
        if data["img_path"].endswith("gt.jpg"):
            continue
        img_path = data["img_path"]
        piping_id = osp.basename(img_path).split('_')[:2]
        piping_id = ' '.join(piping_id)
        if piping_id in test_pipings:
            test_dataset['data_list'].append(data)
        else:
            train_dataset['data_list'].append(data)


    output_test = osp.join(save_dir, 'test_mmengine.json')
    output_train = osp.join(save_dir, 'train_mmengine.json')

    with open(output_test, 'w') as f:
        json.dump(test_dataset,f)

    with open(output_train, 'w') as f:
        json.dump(train_dataset, f)
        


def check_ratio_of_annos(piping_ids: list, ratio:float, total_anno_count: Counter, piping_img_dict:Dict[str, List], img_anno_count_dict: Dict[str, Counter]):
    sampled_cnt = Counter()
    for pid in piping_ids:
        img_list = piping_img_dict[pid]
        for img in img_list:
            sampled_cnt.update(img_anno_count_dict[img])

   
    most_common, least_common = sampled_cnt.most_common()[4], sampled_cnt.most_common()[-29]
    if most_common[1]/least_common[1] > 5:
        print(f"inbalance flag 1: {most_common}, {least_common}")
        return False
    

    for key in sampled_cnt:
        tmp_r = sampled_cnt.get(key)/total_anno_count.get(key)
        print(f"{key} anno ratio :{tmp_r}")
        if tmp_r/ratio < 0.3 or tmp_r > 2:
            print(f"conflict with ratio {ratio}!")
            return False 
    return True

def check_ratio_of_defects(piping_ids: list, ratio:float, total_defect_count: Counter, piping_img_sequence_dict:Dict[str, List], img_sequence_count_dict:Dict[str, Counter]):
    sampled_cnt = Counter()
    for pid in piping_ids:
        img_list = piping_img_sequence_dict[pid]
        for img in img_list:
            # print(img)
            # print(img_sequence_count_dict[img])
            sampled_cnt.update(img_sequence_count_dict[img])

   
    most_common, least_common = sampled_cnt.most_common()[4], sampled_cnt.most_common()[-29]
    if most_common[1]/least_common[1] > 5:
        print(f"inbalance flag2: {most_common} {least_common}")
        return False
    

    for key in sampled_cnt:
        tmp_r = sampled_cnt.get(key)/total_defect_count.get(key)
        print(f"{key} anno ratio :{tmp_r}")
        if tmp_r/ratio < 0.3 or tmp_r > 2:
            print(f"conflict with ratio {ratio}!")
            return False 
    return True

    



    




# split_piping_dataset('/data/ours_annotation/annotations/mmengine_format_LJ_defect.json', '/data/ours_annotation/annotations/image1-2_train_test', [0.9, 0.1], True, None)
input_file = "/data/ours_annotation/annotations/annotations_v3/mmengine_format_v3.json"
output_dir = "/data/ours_annotation/annotations/annotations_v3"

split_piping_dataset(input_file, output_dir, [0.95, 0.05], True, None)
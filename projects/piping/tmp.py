import os 
import json

from mmdet.models.losses import CrossEntropyLoss

from mmengine.runner import LogProcessor
from mmengine.hooks import LoggerHook
from mmdet.visualization import DetLocalVisualizer
from mmdet.engine.hooks import DetVisualizationHook
from mmengine.hooks import LoggerHook

# dir1 = '/data/ours_annotation/images/images4/'
# dir2 = '/data/ours_annotation_ori/images/images4/images4/'

# images1 = set(os.listdir(dir1))
# images2 = set(os.listdir(dir2))

# for img in images1:
#     if img.endswith('xml'):
#         os.remove(os.path.join(dir1, img))

#     if img.endswith('gt.jpg'):
#         os.remove(os.path.join(dir1, img))

# print(sorted(images1-images2))
# print(len(images1-images2))

# print('=============')
# print(images2-images1)

# input_file = '/data/ours_annotation/annotations/annotations_v4/mmengine_format_1225.json'
# with open(input_file) as f:
#     s = json.load(f)


import json
import os
import os.path as osp

filelist_dir = '/data/ours_annotation/annotations/annotations_v3'
filelist = ['train_mmengine_images.txt',
            'test_mmengine_images.txt',
            'train_normal_mmengine_images.txt']
filelist = [os.path.join(filelist_dir, v) for v in filelist]

input_file = '/data/ours_annotation/annotations/annotations_v4/mmengine_format_1225.json'
output_dir = '/data/ours_annotation/annotations/annotations_v4'

with open(input_file) as f:
    # print(f.readline())
    input_dataset = json.load(f)

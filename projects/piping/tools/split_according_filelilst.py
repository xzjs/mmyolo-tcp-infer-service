import json
import os
import os.path as osp

filelist_dir = '/data/ours_annotation/annotations/annotations_v3'
filelist = ['train_mmengine.json',
            'test_mmengine.json',
            'train_normal_mmengine.json']
filelist = [os.path.join(filelist_dir, v) for v in filelist]

input_file = '/data/ours_annotation/annotations/annotations_v4/mmengine_format_1225.json'
output_dir = '/data/ours_annotation/annotations/annotations_v4'

with open(input_file) as f:
    # print(f.readline())
    input_dataset = json.load(f)

output_dataset = dict()
output_dataset['metainfo'] = input_dataset['metainfo']


for file in filelist:
    output_dataset['data_list'] = []
    with open(file) as f:
        tmp_dataset = json.load(f)
        img_list = [v['img_path'] for v in tmp_dataset['data_list']]
        img_list = set(img_list)

    out_img = set()
    for data in input_dataset['data_list']:
        if data['img_path'] in img_list:
            output_dataset['data_list'].append(data)
            out_img.add(data['img_path'])


    for data in tmp_dataset['data_list']:
        if not data['img_path'] in out_img:
            output_dataset['data_list'].append(data)
        

    output_file = osp.join(output_dir, osp.basename(file).replace('_images.txt', '.json'))
    with open(output_file, 'w') as f:
        json.dump(output_dataset, f)




import json
import os 

input_dir = '/data/ours_annotation/annotations/annotations_v3'
input_files = ['train_mmengine.json', 
               'train_normal_mmengine.json', 
               'test_mmengine.json',
               ]

input_files = [os.path.join(input_dir, v) for v in input_files]

for file in input_files:
    with open(file) as f:
        dataset = json.load(f)

    output_file = os.path.splitext(file)[0]+'_images.txt'
    out = open(output_file, 'w')

    data_list = dataset['data_list']
    for d in data_list:
        out.write(d['img_path'])
        out.write('\n')

    out.close()
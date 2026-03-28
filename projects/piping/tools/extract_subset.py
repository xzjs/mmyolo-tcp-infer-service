import json

input_file = '/data/ours_annotation/annotations/annotations_v3/mmengine_format_v3.json'
output_file = '/data/ours_annotation/annotations/mmengine_format_images34.json'

target_subset = ['images4', 'images3']

with open(input_file) as f:
    input_dataset = json.load(f)

output_dataset = dict()
output_dataset['metainfo'] = input_dataset['metainfo']
output_dataset['data_list'] = []

for data in input_dataset['data_list']:
    for t in target_subset:
        if t in data['img_path']:
            output_dataset['data_list'].append(data)
            break 

with open(output_file, 'w') as f:
    json.dump(output_dataset, f)
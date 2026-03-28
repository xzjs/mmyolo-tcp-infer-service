import json
import random


k=256
input_file = "/data/ours_annotation/annotations/mmengine_format.json"
output_file = "/data/ours_annotation/annotations/mmengine_format_filtered_gt.json" 
with open(input_file) as f:
    data = json.load(f) 

print(data.keys())
# data['data_list'] = random.sample(data['data_list'], k=k)
data['data_list'] = [v for v in data['data_list'] if not v['img_path'].endswith('gt.jpg')]

print(len(data['data_list']))

for idx, d in enumerate(data['data_list']):
    d['img_id'] = idx


with open(output_file, 'w') as f:
    json.dump(data, f)
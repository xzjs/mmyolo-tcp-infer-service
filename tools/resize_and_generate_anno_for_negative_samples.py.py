import cv2 
import json
import glob 
import os.path as osp
import os

input_dir = '/data/ours_annotation_ori/images/normal_images'
output_annotation_file = '/data/ours_annotation/annotations/mmengine_format_normal.json'
output_images_dir = '/data/ours_annotation/images/normal_images'
os.makedirs(output_images_dir, exist_ok=True)
output_size = (640, 360)

# resize images
image_list = glob.glob(osp.join(input_dir,'**','*.jpg'), recursive=True)
# for img_path in image_list:
#     im = cv2.imread(img_path)
#     im = cv2.resize(im, output_size)
#     output_img_path = osp.join(output_images_dir, osp.basename(img_path))
#     cv2.imwrite(output_img_path, im)

# generate negative samples' annotation file
dataset = dict()
dataset['metainfo'] = {'classes':[]}
subdir = osp.basename(output_images_dir)
dataset['data_list'] = [{'img_path': osp.join(subdir, osp.basename(img_path)), 
                         'objects': []} for img_path in image_list]

with open(output_annotation_file, 'w') as f:
    json.dump(dataset, f)

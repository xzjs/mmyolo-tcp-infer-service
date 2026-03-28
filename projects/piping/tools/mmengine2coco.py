import json 
import os.path as osp

k = 256
data_root = '/data'
# input_file = f'ours_annotation/annotations/mmengine_format_{k}.json'
# output_file = f'ours_annotation/annotations/coco_format_{k}.json'

# input_file = f'ours_annotation/annotations/mmengine_format_filtered_gt.json'
# output_file = f'ours_annotation/annotations/coco_format_filtered_gt.json'

# input_file = f'ours_annotation/annotations/mmengine_format_LJ_defect.json'
# output_file = f'ours_annotation/annotations/coco_format__LJ_defect.json'

# input_file = f'/data/ours_annotation/annotations/image1-2_defect_LJ_train_test/train_mmengine.json'
# output_file = f"/data/ours_annotation/annotations/image1-2_defect_LJ_train_test/train_coco_format.json"

input_file = f'/data/ours_annotation/annotations/image1-2_defect_LJ_train_test/test_mmengine.json'
output_file = f"/data/ours_annotation_ori/annotations/image1-2_defect_LJ_train_test/test_coco_format.json"

input_file = f'/data/ours_annotation/annotations/image1-2_defect_LJ_train_test/train_mmengine.json'
output_file = f"/data/ours_annotation_ori/annotations/image1-2_defect_LJ_train_test/train_coco_format.json"

input_file = '/data/ours_annotation/annotations/image3_train_test/test_mmengine.json'
output_file = '/data/ours_annotation/annotations/image3_train_test/test_coco_format.json'

input_file = '/data/ours_annotation/annotations/image3_train_test/train_mmengine.json'
output_file = '/data/ours_annotation/annotations/image3_train_test/train_coco_format.json'

input_file = '/data/ours_annotation/annotations/image3_train_test/mmengine_format_images3.json'
output_file = '/data/ours_annotation/annotations/image3_train_test/coco_format_images3.json'

input_file = '/data/ours_annotation_ori/annotations/annotations_v3/test_mmengine.json'
output_file = '/data/ours_annotation_ori/annotations/annotations_v3/test_coco_format.json'

input_file = '/data/ours_annotation_ori/annotations/annotations_v3/train_mmengine.json'
output_file = '/data/ours_annotation_ori/annotations/annotations_v3/train_coco_format.json'

input_file = '/data/ours_annotation/annotations/image3_train_test/mmengine_format_images12.json'
output_file = '/data/ours_annotation/annotations/image3_train_test/coco_format_images12.json'

input_file = '/data/ours_annotation/annotations/mmengine_format_images34.json'
output_file = '/data/ours_annotation/annotations/coco_format_images34.json'

# input_file = '/data/ours_annotation/annotations/annotations_v3/train_normal_mmengine.json'
# output_file = '/data/ours_annotation/annotations/annotations_v3/train_normal_coco_format.json'

# input_file = '/data/ours_annotation/annotations/mmengine_format_images4.json'
# output_file = '/data/ours_annotation/annotations/coco_format_images4.json'

input_file = '/data/ours_annotation/annotations/annotations_v4/train_normal_mmengine.json'
output_file = '/data/ours_annotation/annotations/annotations_v4/train_normal_coco_format_with_level.json'

input_file = '/data/ours_annotation/annotations/annotations_v4/test_mmengine.json'
output_file = '/data/ours_annotation/annotations/annotations_v4/test_coco_format_with_level.json'

if data_root:
    input_file = osp.join(data_root, input_file)
    output_file = osp.join(data_root, output_file)

def convert(input_file, output_file, image_size=None, image_root=None):
    with open(input_file) as f:
        anno_mmdet = json.load(f)

    anno_coco = dict()
    anno_coco['categories'] = []
    for idx, c in enumerate(anno_mmdet['metainfo']['classes']):
        anno_coco['categories'].append({'id': idx, 'name': c})

    print(len(anno_coco['categories']))

    anno_coco['images'] = []
    anno_coco['annotations'] = []

    anno_idx = 0
    sample_categories = set()

    if image_size is None and image_root is None:
        width, height = 640, 320
    elif image_size is not None and image_root is None: 
        width, height = image_size
    else:
        if image_size is not None:
            print("Both `image_size` and `image_root` are set, using the original image size.")
       
        from PIL import Image 

    for img_id, data in enumerate(anno_mmdet['data_list']):
        if image_root:
            img_path = osp.join(image_root, data['img_path'])
            img = Image.open(img_path)
            width, height = img.size # width, height
        anno_coco['images'].append({'id': img_id, 
                                    'file_name': data['img_path'], 
                                    'width':width, 
                                    'height':height})
        for obj in data['objects']:
            anno_coco['annotations'].append({
                'id': anno_idx,
                'image_id': img_id,
                'category_id': obj['defect_name'],
                'area': obj['bncbox']['width']*obj['bncbox']['height']*width*height/10000,
                'bbox': [obj['bncbox']['x']*width/100, 
                         obj['bncbox']['y']*height/100, 
                         obj['bncbox']['width']*width/100, 
                         obj['bncbox']['height']*height/100],
                'bbox_level':  obj['defect_level'],
                'iscrowd' :0
            })
            anno_idx += 1
            sample_categories.add(obj['defect_name'])


    print(f"Number of total categories: {len(sample_categories)}\n \
          Names of categories: {sample_categories}")
    for c in range(len(anno_mmdet['metainfo']['classes'])):
        if c not in sample_categories:
            print(anno_mmdet['metainfo']['classes'][c])

    print(f"Number of categories in dataset: {len(anno_coco['categories'])}")
    print(f"Dumping coco format annotation...")
    with open(output_file, 'w') as f:
        json.dump(anno_coco, f)
    

image_root = '/data/ours_annotation_ori/images/'
image_root = None
convert(input_file, output_file)

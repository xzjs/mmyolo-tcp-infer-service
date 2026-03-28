import json
import copy

defect_annotations = '/data/ours_annotation/annotations/mmengine_format_filtered_gt.json'
lj_annotations = '/data/ours_annotation/annotations/mmengine_format_LJ.json'
output_annotations_ori = '/data/ours_annotation/annotations/mmengine_format_LJ_defect.json'
output_annotations = '/data/ours_annotation/annotations/mmengine_format_LJ_defect_2.json'

# with open(lj_annotations) as f:
#     lj_anno = json.load(f)
#     assert len(lj_anno['metainfo']['classes']) == 1
# with open(defect_annotations) as f:
#     defect_anno = json.load(f)

# # lj as the last category
# defect_anno["metainfo"]["classes"].append(lj_anno['metainfo']['classes'][0])
# new_class_idx = defect_anno["metainfo"]["classes"].index(lj_anno['metainfo']['classes'][0])


# lj_anno_dict = {}
# for ann in lj_anno["data_list"]:
#     # assert len(ann["objects"]) <= 1, f"{ann['objects']}"
#     if len(ann["objects"]) == 0:
#         continue
#     lj_anno_dict[ann['img_path']] = ann['objects']


# for ann in defect_anno['data_list']:
#     img_path = ann["img_path"]
#     new_objects = lj_anno_dict.get(img_path, None)
#     if not new_objects is None:
#         for obj in new_objects:
#             obj["defect_name"] = new_class_idx
        
#         ann["objects"].extend(new_objects)

# with open(output_annotations, 'w') as f:
#     json.dump(defect_anno, f)

def load(anno_file):
    with open(anno_file) as f:
        d = json.load(f)
    return d

def merge_objects(obj_list1, obj_list2, class2_index_map):
    if obj_list1 is None:
        obj_list1 = []
    if obj_list2 is None:
        obj_list2 = []

    output = copy.deepcopy(obj_list1)
    for obj in obj_list2:
        obj['defect_name'] = class2_index_map[obj['defect_name']]
        output.append(obj)
    return output

def merge_annotations(anno_file1, anno_file2, output_file):
    dataset1 = load(anno_file1)
    dataset2 = load(anno_file2)

    classes = copy.copy(dataset1['metainfo']['classes'])
    classes2 = [c for c in dataset2['metainfo']['classes'] if c not in classes]
    classes += classes2

    class2_index_map = {}
    for idx,c in enumerate(dataset2['metainfo']['classes']):
        class2_index_map[idx] = classes.index(c)

    obj_dict1 = {v['img_path']:v['objects'] for v in dataset1['data_list']}
    obj_dict2 = {v['img_path']:v['objects'] for v in dataset2['data_list']}

    img_list = set(list(obj_dict1.keys())+list(obj_dict2.keys()))
    data_list = []
    for img_path in img_list:
        objects = merge_objects(obj_dict1.get(img_path), obj_dict2.get(img_path), class2_index_map)
        data = {'img_path':img_path, 'objects':objects}

        data_list.append(data)

    output_dataset = {'metainfo':{'classes':classes}, 'data_list':data_list}
    with open(output_file, 'w') as f:
        json.dump(output_dataset, f)

# merge_annotations(defect_annotations, lj_annotations, output_annotations)

def test():
    dataset1 = load(output_annotations_ori)
    dataset2 = load(output_annotations)

    # assert len(dataset1['data_list']) == len(dataset2['data_list']), \
    #     f"""length of dataset 1 is {len(dataset1['data_list'])},
    #      while length of dataset 2 is {len(dataset2['data_list'])} """

    obj_dict1 = {v['img_path']:v['objects'] for v in dataset1['data_list']}
    obj_dict2 = {v['img_path']:v['objects'] for v in dataset2['data_list']}

    for d in dataset1['data_list']:
        img_path = d['img_path']
        assert obj_dict1[img_path] == obj_dict2[img_path]


annotation1 = '/data/ours_annotation/annotations/annotations_v3/train_mmengine.json'
annotation2 = '/data/ours_annotation/annotations/mmengine_format_normal.json'

output = '/data/ours_annotation/annotations/annotations_v3/train_normal_mmengine.json'

merge_annotations(annotation1, annotation2, output)


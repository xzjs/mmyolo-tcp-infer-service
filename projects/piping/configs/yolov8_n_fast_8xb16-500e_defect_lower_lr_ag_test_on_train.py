_base_ = ['./yolov8_n_fast_8xb16-500e_defect_lower_lr_ag.py']

# data_root = '/data/ours_annotation_ori/'
data_root='/data/ours_annotation/'
# val_ann_file = 'annotations/image1-2_defect_LJ_train_test/train_coco_format.json'
val_ann_file = 'annotations/coco_format_images4.json'

# val_ann_file = 'annotations/annotations_v3/train_coco_format.json'

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.2),  # NMS type and threshold
    max_per_img=30000)  # Max number of detections of each image

model = dict(
    test_cfg=model_test_cfg
)


class_names = ['破裂', 
               '变形', 
               '腐蚀', 
               '错口', 
            #    '起伏', 
               '脱节', 
               '接口材料脱落', 
               '支管暗接', 
               '异物穿入', 
              '渗漏', 
               '沉积', 
               '结垢',
               '障碍物',
               '残墙、坝根', 
               '树根', 
                # '浮渣', 
            #    '连接'
               ]
metainfo = dict(classes=class_names)
img_scale=(640, 352)
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=(img_scale)),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

train_dataloader = dict(
    
    dataset=dict(
        data_root = data_root,
        ann_file=val_ann_file,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline= test_pipeline,
        metainfo=metainfo
    ))

val_dataloader = dict(
    
    dataset=dict(
        data_root = data_root,
        ann_file=val_ann_file,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ))


val_evaluator = dict(
    ann_file=data_root + val_ann_file,

    )

test_dataloader = val_dataloader
test_evaluator = val_evaluator

import numpy as np



val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox',
    iou_thrs=np.linspace(
                0.05, 0.3, int(np.round((0.3 - 0.05) / .05)) + 1, endpoint=True).tolist(),
    metric_items=['mAP',
                # 'mAP_50',
                # 'mAP_75',
                # 'mAP_s',
                # 'mAP_m',
                # 'mAP_l',
                'AR@100',
                'AR@300',
                'AR@1000',
                'AR_s@1000',
                'AR_m@1000',
                'AR_l@1000'],
    classwise=True
    )
test_evaluator = val_evaluator


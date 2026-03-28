from mmyolo.datasets.yolov5_coco import YOLOv5CocoDataset
from mmyolo.models.detectors.yolo_detector import YOLODetector
from mmyolo.datasets.utils import yolov5_collate
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import _utils
DataLoader = torch.utils.data.DataLoader
from mmengine.registry import init_default_scope


init_default_scope('mmyolo')
affine_scale = 0.5
img_scale = [640, 640]
max_aspect_ratio = 100
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(_scope_='mmyolo',type='LoadAnnotations', with_bbox=True),
    dict(
        _scope_='mmyolo',
        type='Mosaic',
        img_scale=(640, 352),
        pad_val=114.0,
        use_cached = True,
        pre_transform=[dict(type='LoadImageFromFile'),
                       dict(type='LoadAnnotations')]),
    dict(
        _scope_='mmyolo',
        type='YOLOv5RandomAffine',
        max_rotate_degree=10.0,
        max_shear_degree=10.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

class_names = ['破裂',
               '支管暗接',
               '渗漏']

data_root = '/data/ours_annotation/'
train_ann_file = 'annotations/coco_format_256.json'
train_data_prefix = 'images/' 

palettes = [(220, 20, 60), (119, 11, 32), (0, 0, 142)]
num_classes = len(class_names)  # Number of classes for classification
metainfo = dict(
    classes=class_names,
    palette=palettes[:num_classes],
    
)

# train_dataloader = dict(
#     batch_size=2,
#     num_workers=0,
#     persistent_workers=True,
#     pin_memory=False,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     collate_fn=dict(type='yolov5_collate'),
#     dataset=dict(
#         type='YOLOv5CocoDataset',
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file=train_ann_file,
#         data_prefix=dict(img=train_data_prefix),
#         # filter_cfg=dict(filter_empty_gt=False, min_size=32),
#         pipeline=train_pipeline))
dataset = YOLOv5CocoDataset(data_root=data_root,
                            metainfo=metainfo,
                            ann_file=train_ann_file,
                            data_prefix=dict(img=train_data_prefix),
                            pipeline=train_pipeline)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=yolov5_collate)

# for d in dataset:
#     print(d.keys())
#     print(d['inputs'].shape)
#     break

last_stage_out_channels=1024
deepen_factor = 0.25
widen_factor = 0.25
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001) 
strides = [8, 16, 32]

loss_cls_weight = 0.5
loss_bbox_weight = 0.75
loss_dfl_weight = 1.5 / 4

tal_topk = 10  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics

# model = YOLODetector(data_preprocessor=dict(type='YOLOv5DetDataPreprocessor',
#                                             mean=[0., 0., 0.],
#                                             std=[255., 255., 255.],
#                                             bgr_to_rgb=True),
#                     backbone=dict(type='YOLOv8CSPDarknet',
#                                     arch='P5',
#                                     last_stage_out_channels=last_stage_out_channels,
#                                     deepen_factor=deepen_factor,
#                                     widen_factor=widen_factor,
#                                     norm_cfg=norm_cfg,
#                                     act_cfg=dict(type='SiLU', inplace=True)),
#                     neck=dict(
#                                 type='YOLOv8PAFPN',
#                                 deepen_factor=deepen_factor,
#                                 widen_factor=widen_factor,
#                                 in_channels=[256, 512, last_stage_out_channels],
#                                 out_channels=[256, 512, last_stage_out_channels],
#                                 num_csp_blocks=3,
#                                 norm_cfg=norm_cfg,
#                                 act_cfg=dict(type='SiLU', inplace=True)),
#                     bbox_head=dict(
#                                     type='YOLOv8Head',
#                                     head_module=dict(
#                                         type='YOLOv8HeadModule',
#                                         num_classes=num_classes,
#                                         in_channels=[256, 512, last_stage_out_channels],
#                                         widen_factor=widen_factor,
#                                         reg_max=16,
#                                         norm_cfg=norm_cfg,
#                                         act_cfg=dict(type='SiLU', inplace=True),
#                                         featmap_strides=strides),
#                                     prior_generator=dict(
#                                         type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
#                                     bbox_coder=dict(type='DistancePointBBoxCoder'),
#                                     # scaled based on number of detection layers
#                                     loss_cls=dict(
#                                         type='mmdet.CrossEntropyLoss',
#                                         use_sigmoid=True,
#                                         reduction='none',
#                                         loss_weight=loss_cls_weight),
#                                     loss_bbox=dict(
#                                         type='IoULoss',
#                                         iou_mode='ciou',
#                                         bbox_format='xyxy',
#                                         reduction='sum',
#                                         loss_weight=loss_bbox_weight,
#                                         return_iou=False),
#                                     loss_dfl=dict(
#                                         type='mmdet.DistributionFocalLoss',
#                                         reduction='mean',
#                                         loss_weight=loss_dfl_weight)),
#                     train_cfg=dict(
#                         assigner=dict(
#                         type='BatchTaskAlignedAssigner',
#                         num_classes=num_classes,
#                         use_ciou=True,
#                         topk=tal_topk,
#                         alpha=tal_alpha,
#                         beta=tal_beta,
#                         eps=1e-9)),

# )

for d in dataloader:

    if d['data_samples']['bboxes_labels'].numel() == 0:
        continue
    print(d['data_samples']['bboxes_labels'].size())
    print(type(d))
    print(d.keys())
    print(d['data_samples'])
    print(d['inputs'].shape)
    print(d['data_samples'].keys())

    # model(d)
    break

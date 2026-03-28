_base_ = './defects_detection_v3/yolov8_s_fast_8xb16-500e_base.py'
# _base_ = ['./yolov8_s_fast_8xb16-500e_defect_lj.py']

# load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth'
# load_from = '/home/zhuangwj/mmyolo/work_dir_n_freeze4_withoutlj/epoch_155.pth'
# load_from = '/home/zhuangwj/mmyolo/work_dir_n_freeze_stage4_adam_v2_close_mosaic/best_coco_破裂_precision_epoch_180.pth'

# load_from = '/root/piping_autodl_bundle/piping_autodl_bundle/mmyolo/weights/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth'
load_from = '/root/piping_autodl_bundle/piping_autodl_bundle/mmyolo/weights/best_coco_破裂_precision_epoch_55.pth'


deepen_factor = 0.33
widen_factor = 0.25

model = dict(
    backbone=dict(deepen_factor=deepen_factor, 
                  widen_factor=widen_factor,
                #   frozen_stages=4,
                  ),
    neck=dict(deepen_factor=deepen_factor, 
              widen_factor=widen_factor,
              ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))


# train_dataloader = dict(
    # dataset=dict(
    #     pipeline=_base_.train_pipeline_stage2))

base_lr = 0.0001
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

max_epochs = 100  # Maximum training epochs
# Disable mosaic augmentation for final 10 epochs (stage 2)
close_mosaic_epochs = 20

default_hooks = dict(
    param_scheduler=dict(
        max_epochs=max_epochs))

train_cfg = dict(
    max_epochs=max_epochs,
)
 



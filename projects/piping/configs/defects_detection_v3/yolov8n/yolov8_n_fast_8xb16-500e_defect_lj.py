_base_ = '../yolov8_s_fast_8xb16-500e_base.py'

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth'


deepen_factor = 0.33
widen_factor = 0.25

max_epochs = 100

close_mosaic_epochs = 20


val_intervals = 5
# validation intervals in stage 2
val_interval_stage2 = 3



model = dict(
    backbone=dict(deepen_factor=deepen_factor, 
                  widen_factor=widen_factor,
                  ),
    neck=dict(deepen_factor=deepen_factor, 
              widen_factor=widen_factor,
              ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor),
                   loss_cls=dict(
                       _delete_=True,
                       _scope_='mmdet',
                       type='FocalLoss',
                       reduction='none',
                       loss_weight=_base_.loss_cls_weight*10
                   )
                ))

base_lr = 0.01
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# 0.0001 -> 0.0002
ema_momentum = 0.0002


custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=_base_.train_pipeline_stage2)
        )
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_intervals,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        val_interval_stage2)])

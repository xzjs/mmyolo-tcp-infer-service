from mmdet.apis import DetInferencer
import pickle as pkl
from mmengine.registry import init_default_scope

init_default_scope( 'mmyolo')

image_dir = '/data/ours_annotation/images/images3/images'
model_path = '/home/zhuangwj/mmyolo/work_dir_n_freeze_stage4_adam_close_mosaic_train_backbone2/best_coco_破裂_precision_epoch_55.pth'

inferencer = DetInferencer(weights=model_path)

inferencer(image_dir, show=False, batch_size=128, no_save_vis=True, no_save_pred=False, out_dir='image3_results')
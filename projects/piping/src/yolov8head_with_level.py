from typing import List, Sequence, Tuple, Union, Optional

import torch
import torch.nn as nn

import math

from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict

from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)

from mmyolo.registry import MODELS
from mmyolo.models.dense_heads.yolov8_head import YOLOv8HeadModule, YOLOv8Head, multi_apply

from mmengine.structures import InstanceData
from torch import Tensor

@MODELS.register_module()
class YOLOv8HeadWithLevelModule(YOLOv8HeadModule):
    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 num_grades:int = 4,
                 grade_scale:float = 1.0,
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 reg_max: int = 16,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(num_classes=num_classes,
                         in_channels=in_channels,
                         widen_factor=widen_factor,
                         num_base_priors=num_base_priors,
                         featmap_strides=featmap_strides,
                         reg_max=reg_max,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)
        self.num_grades=num_grades+1  # including normal grade
        self.grad_scale=grade_scale

        self._init_level_layers()

    def _init_level_layers(self):
        max_stride = max(self.featmap_strides)
        kernel_strides = [max_stride // stride for stride in self.featmap_strides]
        
        cls_out_channels = max(self.in_channels[0], self.num_classes)
        self.level_embeds = nn.ModuleList()
        for i, stride in enumerate(kernel_strides):
            self.level_embeds.append(ConvModule(
                                        in_channels=self.in_channels[i],
                                        out_channels=cls_out_channels,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1,
                                        norm_cfg=self.norm_cfg,
                                        act_cfg=self.act_cfg))
            
        self.level_pred_head = nn.Sequential(ConvModule(
                                        in_channels=cls_out_channels*len(self.featmap_strides),
                                        out_channels=self.num_classes,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1,
                                        norm_cfg=self.norm_cfg,
                                        act_cfg=self.act_cfg),
                                        nn.AdaptiveAvgPool2d((1, 1)))

        self.ordinal_regression_cutpoints = self._build_clm_thresholds()
        self.link_function = self._build_link_function()

    def _build_clm_thresholds(self):
        thresholds_b = torch.rand((1,))*0.1
        minval = math.sqrt(1/(self.num_grades-2)/2)
        maxval = math.sqrt(1/(self.num_grades-2))
        thresholds_a = torch.rand((self.num_grades-2,))*(maxval-minval)+minval
        self.thresholds_a = nn.Parameter(thresholds_a)
        self.thresholds_b = nn.Parameter(thresholds_b)

        ordinal_regression_cutpoints = torch.cat([self.thresholds_b, self.thresholds_a**2])
        ordinal_regression_cutpoints = torch.cumsum(ordinal_regression_cutpoints, dim=0)
        self.ordinal_regression_cutpoints = nn.Parameter(ordinal_regression_cutpoints)

    def _build_link_function(self, link_func = 'logit'):
        if link_func == 'logit':
            return torch.sigmoid
        else:
            raise NotImplementedError
        
        

    def forward(self, x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Forward features from the upstream network.
        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level classification scores, bbox
            predictions and global predition of class grading.
        """

        assert len(x) == self.num_levels
        detection_outs = multi_apply(self.forward_single, x, self.cls_preds,
                           self.reg_preds)
        '''
        ordinal regression
        '''
        grading_embs = [self.level_embeds[i](x[i]) for i in range(len(x))] # (B, cls_out_channels, H, W)
        grading_embs = torch.cat(grading_embs, dim=1) # (B, cls_out_channels*num_levels, 1, 1)
        grading_preds = self.level_pred_head(grading_embs) # (B, num_grades, 1, 1)

        probs = self.link_function(self.ordinal_regression_cutpoints-grading_preds.squeeze(-1)) # (B, num_classes, num_grades) 
        link_mat = probs[:,:,1:] - probs[:,:,:-1] # (B, num_classes, num_grades-1)
        link_mat = torch.cat([probs[:,:,0], link_mat, probs[:,:,-1]], dim=-1)  # (B, num_classes, num_grades+1)

        eps = 1e-15
        likelihoods = torch.clamp(link_mat, eps, 1 - eps)


        return *detection_outs, grading_preds
    
@MODELS.register_module()
class YOLOv8HeadWithLevel(YOLOv8Head):
    def __init__(self,
                 head_module: ConfigType,
                 grading_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=0.5),
                 loss_bbox: ConfigType = dict(
                     type='IoULoss',
                     iou_mode='ciou',
                     bbox_format='xyxy',
                     reduction='sum',
                     loss_weight=7.5,
                     return_iou=False),
                 loss_dfl=dict(
                     type='mmdet.DistributionFocalLoss',
                     reduction='mean',
                     loss_weight=1.5 / 4),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dfl=loss_dfl,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.grading_module = MODELS.build(grading_module)
        
    def loss_by_feat(self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            grading_preds: Tensor,
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:

        detection_loss = super().loss_by_feat(cls_scores, 
                                              bbox_preds, 
                                              bbox_dist_preds, 
                                              batch_gt_instances, 
                                              batch_img_metas, 
                                              batch_gt_instances_ignore)
        
        grading_loss = self.grading_module.loss_by_feat(grading_preds, batch_img_metas)
        return {**detection_loss, 'grading loss': grading_loss}
        
    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        det_out = self.head_module(x)
        grading_out = self.grading_module(x, with_logit=False)
        return *det_out, grading_out
    
    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        grading_preds: Optional[Tensor] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        
        det_results = super().predict_by_feat(cls_scores,
                                              bbox_preds,
                                              batch_img_metas=batch_img_metas,
                                              cfg=cfg,
                                              rescale=rescale,
                                              with_nms=with_nms)
        grading_results = self.grading_module.predict_by_feat(grading_preds) # e.g. [[(c1, l1), (c2,l2)],[]]

        return {'det_results':det_results, 'grading_results': grading_results}
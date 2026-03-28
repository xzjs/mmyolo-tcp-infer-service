import mmengine
from mmyolo.registry import DATASETS, TASK_UTILS
# from ..registry import DATASETS, TASK_UTILS
from mmdet.datasets import BaseDetDataset
from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import os.path as osp



@DATASETS.register_module()
class PipingDataset(BatchShapePolicyDataset, BaseDetDataset):

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset = {'categories': self.metainfo}
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        This method should return dict or list of dict. Each dict or list
        contains the data information of a training sample. If the protocol of
        the sample annotations is changed, this function can be overridden to
        update the parsing logic while keeping compatibility.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            list or list[dict]: Parsed annotation.
        """
        # raw_data_info['img'] = raw_data_info['img_path']
        for prefix_key, prefix in self.data_prefix.items():
            assert prefix_key in raw_data_info, (
                f'raw_data_info: {raw_data_info} dose not contain prefix key'
                f'{prefix_key}, please check your data_prefix.')
            raw_data_info[prefix_key] = osp.join(prefix,
                                                  raw_data_info[prefix_key])

            raw_data_info['instances'] = []
            raw_data_info['height'] = 360
            raw_data_info['width'] = 640

            for obj in raw_data_info['objects']:
                obj_info = dict()
                obj_info['bbox'] = [obj['bncbox']['x']/100*640, 
                                    obj['bncbox']['y']/100*360,
                                    obj['bncbox']['width']/100*640,
                                    obj['bncbox']['height']/100*360
                ]
                obj_info['bbox_label'] = obj['defect_name']
                obj_info['ignore_flag'] = 0
                raw_data_info['instances'].append(obj_info)
        
        return raw_data_info
    


if __name__ == '__main__':
    data_root = '/mnt/e/ours_annotation'
    ann_file = 'annotations/mmengine_format.json'
    data_prefix = dict(img_path='images/')
    pre_transform = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True)
    ]
    # pre_transform=[]
    d = PipingDataset(data_root=data_root, 
                      ann_file=ann_file, 
                      data_prefix=data_prefix,
                      pipeline=pre_transform)
    for data in d:
        print(data)
        break

    print(d.metainfo)
    print(d.keys())

    
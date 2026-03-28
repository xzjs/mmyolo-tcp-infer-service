from mmyolo.datasets.transforms import LoadAnnotations as YOLO_LoadAnnotations
from mmyolo.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadAnnotations_with_level(YOLO_LoadAnnotations):
    def __init__(self,
                 with_level: bool = True,
                 mask2bbox: bool = False,
                 poly2mask: bool = False,
                 merge_polygons: bool = True,
                 **kwargs):
        super().__init__(mask2bbox=mask2bbox, 
                         poly2mask=poly2mask, 
                         merge_polygons=merge_polygons,
                         **kwargs)
        
        self.with_level = with_level

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        if self.with_level:
            results['gt_level'] = {}
            for instance in results.get('instances', []):
                bbox_label = instance['bbox_label']
                bbox_level = instance['bbox_level']
                if bbox_level > results['gt_level'].get(bbox_label, 0):
                    results['gt_level'][bbox_label] = bbox_level
            results['gt_level'] = list(results['gt_level'].items())
            return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f"with_level={self.with_level}, "
        repr_str += f'mask2bbox={self.mask2bbox}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str
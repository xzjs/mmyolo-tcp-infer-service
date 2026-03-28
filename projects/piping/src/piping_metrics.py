from mmyolo.registry import METRICS
from mmengine.evaluator import BaseMetric
from typing import Sequence, Dict, List
from mmengine.logging import MMLogger
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

@METRICS.register_module()
class GradeMetric(BaseMetric):
    def __init__(self, 
                 num_classes:int, 
                 num_grades:int,
                 classnames: Sequence[List] = None,
                 collect_device = 'cpu', 
                 prefix = None, 
                 collect_dir = None):
        super().__init__(collect_device, prefix, collect_dir)
        self.num_classes = num_classes
        self.num_grades = num_grades
        if not classnames is None:
            assert len(classnames) == num_classes, "Length of `classnames` must be equal to `num_classes`"
            self.classnames = classnames
        else:
            self.classnames = [f"class_{ci}" for ci in range(self.num_classes)] 
    def process(self, data_batch: dict, data_samples: Sequence[dict])->None:
        for data_sample in data_samples:
            gt_level = data_sample['gt_level']
            pred_level = data_sample['pred_level']
            self.results.append((gt_level, pred_level))

    def compute_metrics(self, results:list)->Dict[str, float]:
        # logger: MMLogger = MMLogger.get_current_instance()
        acc = self._calucate_accuracy(results)
        singlelabel_acc = self._calculate_singlelabel_classification_accuracy(results)
        confusion_matrix = self._calucate_confusion_matrix(results)
        qwk_results = self._calculate_qwk(confusion_matrix)


        # logger.info('====================================================')
        # logger.info(f"\ncls_acc: {acc['cls acc']}, level_acc: {acc['level acc']}")
        # # logger.info(f'results: {results}')
        # logger.info('====================================================')
        # logger.info('\n')
        # logger.info(confusion_matrix)
        # logger.info('====================================================')
        # logger.info(qwk_results)

        return {**acc, **singlelabel_acc ,'confusion_matrix': confusion_matrix, **qwk_results}

    def _calucate_accuracy(self, results:list)->Dict[str, float]:
        n_cls_correct = 0
        n_level_corect = 0
        n_samples = len(results)
        n_defects = 0
        for gt_level, pred_level in results:
            if len(pred_level) == 0:
                if len(gt_level) == 0:
                    n_cls_correct += 1
                else:
                    n_defects += 1
                continue
            if len(gt_level) == 0:
                continue
            n_defects += 1
            gt_cls = list(zip(*gt_level))[0]
            pred_cls = list(zip(*pred_level))[0]

            for c in pred_cls:
                if c in gt_cls:
                    n_cls_correct += 1
                    break 

            for l in pred_level:
                if l in gt_level:
                    n_level_corect += 1
                    break
        cls_acc = n_cls_correct/n_samples
        level_acc = n_level_corect/n_defects
        return {'cls acc': cls_acc, 'level acc': level_acc}
    
    def _get_subclass_id(self, class_grade):
        return class_grade[0]*(self.num_grades+1) + class_grade[1]
    
    def _calucate_confusion_matrix(self, results):
        confusion_matrix = [np.zeros((self.num_grades+1, self.num_grades+1)) for _ in range(self.num_classes)]

        for gt, pred in results:
            gt = dict(gt)
            pred = dict(pred)

            for ci in range(self.num_classes):
                gl = gt.get(ci, 0)
                pl = pred.get(ci, 0)
                confusion_matrix[ci][gl, pl] += 1
        # confusion_matrix /= len(results)
        # colnames = [(c, g) for c in range(self.num_classes) for g in range(self.num_grades+1)]
        colnames = [g for g in range(self.num_grades+1)]
        # if self.classnames:
        #     colnames = [(self.classnames[c], g) for (c,g) in colnames]
        # confusion_matrix = pd.DataFrame(data=confusion_matrix,index=colnames, columns=colnames)
        confusion_matrix = [pd.DataFrame(data=m, index=colnames, columns=colnames) for m in confusion_matrix]
        confusion_matrix = dict(zip(self.classnames, confusion_matrix))
     
        return confusion_matrix

    def _draw_confusion_matrix(self, confusion_matrix):
        plt.figure(figsize=(10, 6), dpi=100)
        sns.heatmap(confusion_matrix, 
                    # cmap='OrRd',
                    cmap='RdPu',
                    # cmap='coolwarm',
                    # cmap = sns.diverging_palette(0, 0, 90, 60, as_cmap=True),
                    annot=True, 
                    fmt='.2f',
                    # xticklabels=list(confuse_matrix.columns),
                    # yticklabels=list(confuse_matrix.index)
                    xticklabels=True,
                    yticklabels=True,
                    # vmax=0.5,
                    # vmin=-1
                    )
        
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    def _calculate_qwk(self, confusion_matrix):
        def _build_cost_matrix(num_grades):
            num_grades += 1
            cost_matrix = np.zeros((num_grades, num_grades))
            gt = np.arange(num_grades).reshape((num_grades, 1))
            pred = np.arange(num_grades).reshape((1, num_grades))
            cost_matrix = (gt - pred) ** 2 /(num_grades-1)**2
            return cost_matrix
        cost_matrix = _build_cost_matrix(self.num_grades)
        result_list = {}
        for ci in range(self.num_classes):
            c = confusion_matrix[self.classnames[ci]].values
            s = np.sum(c)
            a = np.sum(c * cost_matrix) 
            q1 = np.sum(c, axis=1)
            q2 = np.sum(c, axis=0) 
            eps = 10-6
            q = (q1[:,None] * q2[None,:])/(s+eps)
            q = np.sum(cost_matrix*q)
            ret = 1 - a/(q+eps)
            if self.classnames:
                result_list[self.classnames[ci]] = ret 
            else:
                result_list[f"class_{ci}"] = ret
        return result_list
    
    def _calculate_singlelabel_classification_accuracy(self, results):
        class_correct, level_correct, total_single_class_samples = 0,0,0
        for gt_level, pred_level in results:
            if len(gt_level) != 1:
                continue 
            total_single_class_samples += 1
      
            pred_class = [c for (c, l) in pred_level]

            if gt_level[0][0] in pred_class:
                class_correct += 1

            if gt_level in pred_level:
                level_correct += 1

        return {'singlelabel cls acc': class_correct/total_single_class_samples,
                'singlelabel level acc': level_correct/total_single_class_samples}
        

if __name__ == '__main__':
    # m = GradeMetric(1, 3, [])
    # confusion_matrix = np.array([[0, 0,0,1],
    #                               [0, 0, 1, 0],
    #                               [0, 1, 0, 0],
    #                               [1, 0, 0, 0]])
    # result_list = m._calculate_qwk(confusion_matrix)
    # print(result_list)

    m1 = GradeMetric(2, 3, [])
    datasamples = [{'gt_level':[], 'pred_level':[(0,3)]},
                   {'gt_level':[(0,3),(1,1)], 'pred_level':[(1,1)]},
                   {'gt_level':[(0,1),(1,2)], 'pred_level':[(0,2),(1,2)]},
                   {'gt_level':[(0,2),(1,3)], 'pred_level':[(0,1),(1,3)]}
                  ]
    
    m1.process(None, datasamples)

    m1.compute_metrics(m1.results)

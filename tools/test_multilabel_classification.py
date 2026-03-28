import argparse
import os
import os.path as osp

from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config
import torch
from torch.utils.data import DataLoader
from mmyolo.registry import DATASETS, DATA_SAMPLERS
from mmengine.registry import FUNCTIONS, RUNNERS
from mmengine.registry import init_default_scope

from mmpretrain.evaluation import MultiLabelMetric
from mmengine.evaluator import Evaluator

from mmpretrain.structures import DataSample 
from mmdet.structures import DetDataSample

from functools import partial
import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMYOLO test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')

    args = parser.parse_args()
    return args


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    print(args.config, args.checkpoint)

    # load config
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmyolo'))

    cfg.load_from = args.checkpoint

    dataloader_cfg = cfg.get('test_dataloader')
    # dataset_cfg = dataloader_cfg.pop('dataset')

    # # print(cfg.get('custom_imports'))
    # # print(dataset)

    # dataset = DATASETS.build(dataset_cfg)

    # sampler_cfg = dataloader_cfg.pop('sampler')
    # sampler = DATA_SAMPLERS.build(
    #             sampler_cfg,
    #             default_args=dict(dataset=dataset))
    
    # collate_fn_cfg = dataloader_cfg.pop('collate_fn',
    #                                         dict(type='pseudo_collate'))
    # if isinstance(collate_fn_cfg, dict):
    #     collate_fn_type = collate_fn_cfg.pop('type')
    #     if isinstance(collate_fn_type, str):
    #         collate_fn = FUNCTIONS.get(collate_fn_type)
    #     else:
    #         collate_fn = collate_fn_type
    #     collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
    # elif callable(collate_fn_cfg):
    #     collate_fn = collate_fn_cfg
    # else:
    #     raise TypeError(
    #         'collate_fn should be a dict or callable object, but got '
    #         f'{collate_fn_cfg}')
    

    # dataloader = DataLoader(**dataloader_cfg, dataset=dataset, sampler=sampler)

    # for d in dataset:
    #     print(d)
    #     break

    cfg.work_dir = './work_dirs'
    runner = RUNNERS.build(cfg)
    dataloader = runner.build_dataloader(dataloader_cfg)
    
    runner.test()

    min_thr = 0.1
    max_thr = 0.5
    step_thr = 0.1
    metrics_list = [MultiLabelMetric(thr=v, average=None) for v in np.linspace(
                min_thr, max_thr, int(np.round((max_thr - min_thr) / step_thr)) + 1, endpoint=True)]
    evaluator = Evaluator(metrics=[MultiLabelMetric(topk=1, average=None),
                                   MultiLabelMetric( topk=2, average=None),
                                ] + metrics_list
                                  )
    classnames = dataloader.dataset.metainfo['classes']
    num_classes = len(classnames)
    with torch.no_grad():
        for idx, data_batch in enumerate(tqdm.tqdm(dataloader)):
            
            outputs = runner.model.test_step(data_batch)
            # print(len(outputs))
            # print(outputs[0])
            outputs = list(map(partial(det2cls, num_classes=num_classes), outputs))
            # break 
            evaluator.process(data_samples=outputs, data_batch=data_batch)

    metrics = evaluator.evaluate(len(dataloader.dataset))
    metrics_items = ['precision', 'recall']

    for item in metrics:
        for k in metrics.keys():
            if item in k:
                metric = k 

        print(f"================{metric}==================")
        for c, s in zip(classnames, metrics[metric]):
            print(c, s)


def det2cls(detdata: DetDataSample, num_classes)->DataSample:
    img_meta = detdata.metainfo 
    img_meta['num_classes'] = num_classes

    clsdata = DataSample(metainfo=img_meta)

    det_gt_instances = detdata.gt_instances
    gt_labels = set(det_gt_instances.labels.tolist())

    clsdata.set_gt_label(list(gt_labels))

    pred_scores = torch.zeros(num_classes)
    det_pred_instances = detdata.pred_instances
    for s, l in zip(det_pred_instances.scores, det_pred_instances.labels):
        if s > pred_scores[l]:
            pred_scores[l] = s 

        # pred_scores[l] += s.cpu() 

    pred_scores.clamp_(max=1)


    clsdata.set_pred_score(pred_scores)
    return clsdata








if __name__ == '__main__':
    main()
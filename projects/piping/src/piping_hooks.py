import logging
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

import numpy as np

@HOOKS.register_module()
class ThresholdsHook(Hook):
    def __init__(self, 
                 interval: int = 10):
        
        self.interval = interval
    
    def after_train_iter(self, runner, batch_idx, data_batch = None, outputs = None):
        if self.every_n_inner_iters(batch_idx, self.interval):
            thresh_a = runner.model.bbox_head.grading_module.thresholds_a.detach().cpu().numpy()
            thresh_b = runner.model.bbox_head.grading_module.thresholds_b.detach().cpu().numpy()

            threshs = np.concatenate([thresh_b, thresh_a**2], axis=1)
            threshs = np.cumsum(threshs, axis=1)
         
            runner.logger.info(f"\nthreshold parameters:\n{threshs}")
            # thresh_a_grad = runner.model.bbox_head.grading_module.thresholds_a.grad.norm().item()
            # thresh_b_grad = runner.model.bbox_head.grading_module.thresholds_b.grad.norm().item()
            # runner.logger.info(f"threshold parameters gradient norm:\n \
            #                    threshold_a_grad_norm={thresh_a_grad},\n \
            #                    threshold_b_grad_norm={thresh_b_grad}")
            
            for (name, data) in runner.model.bbox_head.named_parameters():
                if data.grad is not None:
                    print(f"{name=}, data_norm={data.norm().item()}, grad_norm={data.grad.norm().item()}")

                else:
                    # print(f"{name=}, data_norm={data.norm().item()}, grad_norm={None}")
                    pass

                # if 'thresholds' in name:
                #     for i in range(len(runner.optim_wrapper.param_groups)):
                #         all_params = [id(p) for p in runner.optim_wrapper.param_groups[i]['params']]
                #         # print("--------------------")
                #         # print(f"param_group {i}")
                #         # for k, v in runner.optim_wrapper.param_groups[i].items():
                #         #     if k == 'params':
                #         #         continue
                            
                #         #     print(f"{k}:{v}")
                #         #     print("-----------------")
                #         f = id(data) in all_params
                #         if f:
                #             print(f"目标参数{name}在优化器中{i}组")
                #             break
                #         else:
                #             print(f"目标参数{name}不在优化器中{i}组")
                    

            # print(1)
            # for group in runner.optimizer.param_groups:
            #     for p in group['params']:
            #         print(p)


           
@HOOKS.register_module()
class ParametersHook(Hook):
    def __init__(self, interval=1):
        super().__init__()
        self.interval = 1

    def after_train_iter(self, runner, batch_idx, data_batch = None, outputs = None):
        if self.every_n_inner_iters(batch_idx, self.interval):
            norms = {}
            for name, param in runner.model.named_parameters(): 
                if param.requires_grad:   # 仅记录可训练参数 
                    norms[f'param_norm/{name}'] = param.data.norm(2).item() 
                    # print(f'param_norm/{name}:{param.data.norm(2).item()}')
        
            # 将结果写入日志 
            runner.message_hub.update_scalars(norms) 

@HOOKS.register_module()
class VisualizeConfusionMatrix(Hook):
    def after_val_epoch(self, runner, metrics = None):
        
        if (isinstance(runner._train_loop, dict)
                    or runner._train_loop is None):
            epoch = 0
        else:
            epoch = runner.epoch

        confusion_matrix = metrics['grade/confusion_matrix']
        num_classes = runner.model.bbox_head.grading_module._num_classes
        num_grades = runner.model.bbox_head.grading_module._num_grades

        import matplotlib.pyplot as plt
        import seaborn as sns
        import io
        from PIL import Image
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
        
        for title, matrix in confusion_matrix.items():            
            plt.figure(figsize=(10, 6), dpi=100)
            sns.heatmap(matrix, 
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
            plt.title(title)
        
            buffer = io.BytesIO()
            plt.savefig(buffer, format='jpg')  
            buffer.seek(0)  
    
            image = Image.open(buffer)
            image = np.array(image)
    
            buffer.close()
    
            runner.visualizer.add_datasample(
                name = f"confusion_matrix_epoch_{epoch}",
                image = image
            )
        
    
@HOOKS.register_module()
class VisualizeGradients(Hook):
    def __init__(self, interval=10):
        super().__init__()
        self.interval = interval

    def before_train(self, runner) -> None:
        wandb = runner.visualizer.get_backend('WandbVisBackend').experiment
        wandb.watch(runner.model, log='all', log_freq=self.interval)
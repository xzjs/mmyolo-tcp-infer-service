from mmyolo.registry import MODELS

from mmcv.cnn import ConvModule
import math
from typing import List, Sequence, Tuple, Union
from mmengine.model import BaseModule
import torch
import torch.nn as nn
from torch import Tensor
from mmyolo.models.utils import make_divisible

from mmdet.utils import OptMultiConfig, ConfigType

@MODELS.register_module()
class GradingModule(BaseModule):
    def __init__(self,
                 num_classes:int,
                 num_grades:int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 link_func='logit'):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels
        self.num_levels = len(in_channels)
        self._num_classes = num_classes
        self._num_grades = num_grades+1 # including normal grade
        self.featmap_strides = featmap_strides
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.cost_matrix = None
        self.link_func = link_func
        self._init_layers()
        

    def _init_layers(self):
        max_stride = max(self.featmap_strides)
        kernel_strides = [max_stride // stride for stride in self.featmap_strides]
        
        cls_out_channels = max(self.in_channels[0], self._num_classes)
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
                                        out_channels=self._num_classes,
                                        kernel_size=3,
                                        stride=stride,
                                        padding=1,
                                        norm_cfg=self.norm_cfg,
                                        act_cfg=self.act_cfg),
                                        nn.AdaptiveMaxPool2d((1, 1)))

        self._build_clm_thresholds()
        self.link_function = self._build_link_function()

    def _build_clm_thresholds(self):
        thresholds_b = torch.rand((self._num_classes, 1))*0.1
        minval = math.sqrt(1/(self._num_grades-2)/2)
        maxval = math.sqrt(1/(self._num_grades-2))
        thresholds_a = torch.rand((self._num_classes, self._num_grades-2))*(maxval-minval)+minval
        self.thresholds_a = nn.Parameter(thresholds_a)
        self.thresholds_b = nn.Parameter(thresholds_b)


    def _build_link_function(self):
        if self.link_func == 'logit':
            return torch.sigmoid
        elif self.link_func == 'clog':
            def clog(x):
                x = -torch.exp(x)
                x = torch.exp(x)
                return 1-x 
            return clog
        else:
            raise NotImplementedError
        

    def forward(self, x: Tuple[torch.Tensor], with_logit = False) -> Tuple[torch.Tensor]:
        assert len(x) == self.num_levels
        grading_embs = [self.level_embeds[i](x[i]) for i in range(len(x))] # (B, cls_out_channels, H, W)
        grading_embs = torch.cat(grading_embs, dim=1) # (B, cls_out_channels*num_levels, 1, 1)
        grading_preds = self.level_pred_head(grading_embs).squeeze(-1) # (B, num_classes, 1)

        ordinal_regression_cutpoints = torch.cat([self.thresholds_b, self.thresholds_a**2], axis = 1)  # (num_classes, num_grades)
        ordinal_regression_cutpoints = torch.cumsum(ordinal_regression_cutpoints, dim=1) # (num_classes, num_grades)
        probs = self.link_function(ordinal_regression_cutpoints[None, :, :] - grading_preds) # (B, num_classes, num_grades) 
        likelihoods = probs[:,:,1:] - probs[:,:,:-1] # (B, num_classes, num_grades-1)
        likelihoods = torch.cat([probs[:,:,0][:, :, None], likelihoods, 1-probs[:,:,-1][:,:,None]], dim=-1)  # (B, num_classes, num_grades+1)

       
        eps = 1e-15
        likelihoods = torch.clamp(likelihoods, eps, 1 - eps)
        if with_logit:
            return likelihoods, grading_preds 
        else:
            return likelihoods
    
    def predict_by_feat(self, grade_preds: Tensor):

  
        batchsize = grade_preds.shape[0]
        grade_preds = torch.argmax(grade_preds, dim=-1)
       
        batch_id, cls = torch.where(grade_preds)
        grade_preds = grade_preds[batch_id, cls]
        ret = [[] for _ in range(batchsize)] 
        batch_id = batch_id.cpu().numpy()
        cls = cls.cpu().numpy()
        grade_preds = grade_preds.cpu().numpy()
        for (b, c, g) in zip(batch_id, cls, grade_preds):
            ret[b].append((c, g))
        return ret 
    
    def loss_by_feat(self, grade_preds: Tensor, batch_img_metas: Sequence[dict]):
        grade_gt = self._preprocess_gt_grades(batch_img_metas, device=grade_preds.device) # (B, num_classes, num_grades)
        batchsize = grade_gt.shape[0]
        if self.cost_matrix is None:
            self.cost_matrix = self._construct_cost_matrix(self._num_grades, device=grade_preds.device)
        loss = self.qwk_loss(grade_preds, grade_gt)
        loss /= self._num_classes
        loss *= batchsize
        loss *= 1
        return loss

    def qwk_loss(self, grade_preds: Tensor, grade_gt: Tensor):
        grade_preds = grade_preds.permute(1, 0, 2) # (num_classes, B, num_grades)
        grade_gt = grade_gt.permute(1, 0, 2) # (num_classes, B, num_grades)

        costs = self.cost_matrix[torch.where(grade_gt)[-1]] #(num_classes*B, num_grades)
        costs = costs.reshape(grade_preds.shape) # (num_classes, B, num_grades)

        numerator = torch.sum(costs*grade_preds, dim=[-2,-1])

        sum_prob = torch.sum(grade_preds, dim=1) # (num_classes, num_grades)
        nsamples_per_level = torch.sum(grade_gt, dim=1) # (num_classes, num_grades)
        nsamples = torch.sum(nsamples_per_level, dim=-1) # (num_classes,)

        epsilon = 10e-9
        a = torch.matmul(self.cost_matrix[None,:,:], sum_prob[:,:,None]).squeeze(-1) # (num_classes, num_grades) 
        a = nsamples_per_level/(nsamples[:,None]+epsilon)*a # (num_classes, num_grades)

        denominator = torch.sum(a, axis=-1)+epsilon # (num_classes,)
        loss = torch.sum(numerator/denominator)
                
        return loss

    def _preprocess_gt_grades(self, batch_img_metas: Sequence[dict], device):
        gt_grades = []
        for img_meta in batch_img_metas:
            gt_level = img_meta['gt_level']
            gt_grades_tensor = torch.zeros((1, self._num_classes, self._num_grades), device=device)
            gt_level = [l for l in gt_level if l[1] > 0]
            if len(gt_level) > 0:
                class_index, grade_index = list(zip(*gt_level))
                gt_grades_tensor[:, class_index, grade_index] = 1
            gt_grades_tensor[:,:,0] = 1 - torch.sum(gt_grades_tensor, dim=-1)
            gt_grades.append(gt_grades_tensor)

        gt_grades = torch.cat(gt_grades, dim=0)
        assert torch.all(gt_grades.sum(axis=-1)==1)
        return gt_grades
    
    def _construct_cost_matrix(self, num_grades, device):
        cost_matrix = torch.tensor(range(num_grades), dtype=torch.float32, device=device)
        cost_matrix = cost_matrix.view(1, -1).repeat(num_grades, 1) - cost_matrix.view(-1, 1).repeat(1, num_grades)
        cost_matrix = cost_matrix**2/(num_grades-1)**2
        return cost_matrix


if __name__ == '__main__':
    # one class 
    gm = GradingModule(num_classes=1, num_grades=4, in_channels=[8, 16, 32])

    # print("state dict:", list(gm.state_dict()))
    
    # for name, param in gm.named_parameters():
    #     print(name, param.size())


    batchsize = 2
    x = [torch.rand((batchsize, 8, 44, 44)), torch.rand((batchsize, 16, 22, 22)), torch.rand((batchsize, 32, 11, 11))]
    img_meta_info = [{'gt_level':[(0, 2)]}, 
                     {'gt_level':[]}
                    ]
    
    pred = gm(x)
    loss = gm.loss_by_feat(pred, img_meta_info)
    print(loss)

    # multiple classes
    batchsize = 3
    x = [torch.rand((batchsize, 8, 44, 44)), torch.rand((batchsize, 16, 22, 22)), torch.rand((batchsize, 32, 11, 11))]
    gm = GradingModule(num_classes=3, num_grades=4, in_channels=[8, 16, 32])
    gm.to('cuda')
    x = [xx.to('cuda') for xx in x]
    img_meta_info = [{'gt_level':[(1, 3)]}, 
                     {'gt_level':[(1, 3), (2, 2)]},
                     {'gt_level':[]}
                    ]
    
    pred = gm(x)
    loss = gm.loss_by_feat(pred, img_meta_info)
    print(loss, loss.device)


    batchsize = 16
    num_classes = 3 
    num_grades = 4

    # test training
    def mock_data_loader(batchsize, num_classes, num_grades, device):
        torch.manual_seed(0)
        for i in range(1):
            x = [torch.rand((batchsize, 8, 44, 44), device=device), 
                 torch.rand((batchsize, 16, 22, 22), device=device), 
                 torch.rand((batchsize, 32, 11, 11), device=device)]


            img_meta_info = []

            for j in range(batchsize):
                gt_level = []
                for k in range(num_classes):
                    randlevel = torch.randint(0, num_grades+1, (1,)).item()
                    if randlevel > 0:
                        gt_level.append((k, randlevel))
                img_meta_info.append({'gt_level':gt_level})

            yield x, img_meta_info

 

    optimizer = torch.optim.Adam(gm.parameters(), lr=0.01)

    def train_one_epoch(epoch_index):
        running_loss = 0.
        last_loss = 0.

        for idx, data in enumerate(mock_data_loader(batchsize, num_classes, num_grades, device='cuda')):
            feats, img_meta_info = data

            optimizer.zero_grad()

            loss = gm.loss_by_feat(gm(feats), img_meta_info)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if idx % 10 == 0:
                last_loss = running_loss / 1 # loss per batch
                # print('  batch {} loss: {}'.format(idx + 1, last_loss))
                running_loss = 0.

        return last_loss
    
    def validation(epoch_index):
        running_vloss = 0.0 
        gm.eval()

      
        with torch.no_grad():
            for i, vdata in enumerate(mock_data_loader(batchsize, num_classes, num_grades, device='cuda')):
                vinputs, vimg_meta_info = vdata
                voutputs, logits = gm(vinputs, with_logit=True)
                vloss = gm.loss_by_feat(voutputs, vimg_meta_info)
                running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        gm.train(True)
        return avg_vloss, voutputs, logits
    
    def evaluation(epoch_idx):
        gm.eval()
        with torch.no_grad():
            for i, vdata in enumerate(mock_data_loader(batchsize, num_classes, num_grades, device='cuda')):
                vinputs, vimg_meta_info = vdata
                voutputs = gm(vinputs)
                
                preds = gm.predict_by_feat(voutputs)
                voutputs = torch.argmax(voutputs, dim=-1)
                
                print(voutputs, preds, vimg_meta_info)

                vimg_meta_info = [v['gt_level'] for v in vimg_meta_info]

                assert preds == vimg_meta_info
                break

    EPOCHS = 300
    epoch_number = 0

    best_vloss = 1_000_000.

    gm.train(True)
    for epoch in range(EPOCHS):
        # print('EPOCH {}:'.format(epoch + 1))
        # Make sure gradient tracking is on, and do a pass over the data
        

        avg_loss = train_one_epoch(epoch)
        

        if epoch % 100 == 99:
            avg_vloss, voutputs, logits = validation(epoch)
            
        
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            print(gm.thresholds_a, gm.thresholds_b)
            # print(logits)


        if avg_loss < best_vloss:
            best_vloss = avg_loss
            # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            # torch.save(gm.state_dict(), model_path)
        
    evaluation(epoch)

    from torch.utils.data import Dataset, DataLoader
    import mmyolo.piping_metrics

    class MockDataset(Dataset):
        def __init__(self,  num_classes, num_grades, length, device):
            self.num_classes = num_classes
            self.num_grades = num_grades
            self.length = length
            self.device = device
            torch.manual_seed(0)
            self.data = []
            for i in range(length):
                x = [torch.rand((1, 8, 44, 44), device=self.device), 
                     torch.rand((1, 16, 22, 22), device=self.device), 
                     torch.rand((1, 32, 11, 11), device=self.device)]
            
                gt_level = []
                for k in range(num_classes):
                    randlevel = torch.randint(0, num_grades+1, (1,)).item()
                    if randlevel > 0:
                        gt_level.append((k, randlevel))
                img_meta_info = {'gt_level':gt_level}
                self.data.append((x, img_meta_info))
            
        def __len__(self):
            return self.length
        def __getitem__(self, idx):
            return self.data[idx]
        
    def my_collate_fn(batch):
        x, img_meta_info = zip(*batch)
        x = [torch.cat([xx[i] for xx in x], dim=0) for i in range(len(x[0]))]
        return x, img_meta_info
    
    train_dataloader = DataLoader(batch_size=batchsize, 
                                  shuffle=True, 
                                  dataset=MockDataset(num_classes, num_grades, 16, 'cuda'),
                                  collate_fn=my_collate_fn)
    
    val_dataloader = DataLoader(batch_size=batchsize,
                                shuffle=False,
                                dataset=MockDataset(num_classes, num_grades, 16, 'cuda'),
                                collate_fn=my_collate_fn)
    
    from mmengine.model import BaseModel 
    class GMModel(BaseModel):
        def __init__(self):
            super().__init__()
            self.gm = GradingModule(num_classes, num_grades, in_channels=[8, 16, 32])

        def forward(self, inputs, data_samples = None, mode = 'tensor'):
            x = self.gm(inputs)
            if mode == 'loss':
                return {'level losss': self.gm.loss_by_feat(x, data_samples)}
            elif mode == 'predict':
                pred_levels = self.gm.predict_by_feat(x)
                for data, p in zip(data_samples, pred_levels):
                    data['pred_level'] = p
                return data_samples
    
    from torch.optim import SGD 
    from mmengine.runner import Runner 
    from mmyolo.piping import piping_metrics

    # runner = Runner(
    #     model=GMModel(),
    #     work_dir='./work_dirs',
    #     train_dataloader=train_dataloader,
    #     val_dataloader=val_dataloader,
    #     optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.1)),
    #     train_cfg=dict(by_epoch=True, max_epochs=1000, val_interval=100),
    #     val_cfg=dict(),
    #     val_evaluator=dict(type='GradeMetric',_scope_='mmyolo', num_classes=num_classes, num_grades=num_grades),
    # )

    # runner.train()

    model = GMModel()
    for name,v in model.named_modules():
        print(name)
        print(hasattr(v, 'thresholds_a'))
        if isinstance(v, nn.Parameter):
            print(v)


        

    

    


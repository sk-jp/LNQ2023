# from collections import OrderedDict
import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from monai.inferers import SliceInferer, SlidingWindowInferer, SlidingWindowSplitter
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric, SurfaceDistanceMetric

from fix_model_state_dict import fix_model_state_dict
from get_optimizer import get_optimizer
from get_loss import get_loss


class LightningModule(pl.LightningModule):
    def __init__(self, cfg, post_transforms=None):
        super(LightningModule, self).__init__()
        self.cfg = cfg
        self.lossfun = get_loss(cfg.Loss, train=True)
        self.lossfun_valid = get_loss(cfg.Loss, train=False)
        self.txt_logger = cfg.txt_logger
        self.post_transforms = post_transforms

        self.training_step_outputs = []        
        self.validation_step_outputs = []

        self.valid_metrics = ['dice', 'assd']

        self.valid_flag = False

        if cfg.Model.arch == 'flexbile_unet':
            from monai.networks.nets import FlexibleUNet
            self.model = FlexibleUNet(**cfg.Model.params)
        else:
            raise ValueError(f'{cfg.Model.arch} is not supported.')
        
        if cfg.Model.pretrained != 'None':
            # Load pretrained model weights
            print(f'Loading: {cfg.Model.pretrained}')
            checkpoint = torch.load(cfg.Model.pretrained, map_location='cpu')
            state_dict = checkpoint['state_dict']
            state_dict = fix_model_state_dict(state_dict)
            self.model.load_state_dict(state_dict)

        # inferer
        sw_batch_size = self.cfg.Data.dataset.sliding_window_batch_size
        if self.cfg.Data.dataset.sampling_type == "3d":
            self.inferer = SlidingWindowInferer(roi_size=self.cfg.Data.dataset.patch_size,
                                                sw_batch_size=sw_batch_size)
        elif self.cfg.Data.dataset.sampling_type == "2.5d_random":
            self.inferer = SlidingWindowSplitter(patch_size=self.cfg.Data.dataset.sliding_window_size,
                                                 offset=[0, 0, -1],
                                                 overlap=[0, 0, 1])
        else:
            self.inferer = SliceInferer(roi_size=(-1, -1),
                                        sw_batch_size=sw_batch_size,
                                        spatial_dim=2)
        
        self.post_pred = AsDiscrete(argmax=True, to_onehot=cfg.Model.params.out_channels, dim=1)
        self.post_label = AsDiscrete(to_onehot=cfg.Model.params.out_channels, dim=1)

        self.valid_metrics_fun = dict()
        self.valid_metrics_buf = dict()
        self.valid_metrics_fun['dice'] = DiceMetric(include_background=False, 
                                                    reduction="mean",
                                                    num_classes=self.cfg.Data.dataset.num_classes)  
        self.valid_metrics_fun['assd'] = SurfaceDistanceMetric(include_background=False,
                                                               symmetric=True)
        for metric in self.valid_metrics:
            self.valid_metrics_buf[metric] = list()

    def forward(self, x):
        y = self.model(x)
        return y

    def on_train_epoch_start(self):
        # clear up GPU memory
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()                 

    def training_step(self, batch, batch_idx):        
        image = batch["image"]
        label = batch["label"]
#        path = batch["filepath"]

        if self.cfg.Data.dataset.sampling_type == "2.5d_random":
            # extract the center slice
            slice_idx = self.cfg.Data.dataset.num_slices // 2
            label = label[:, slice_idx, :, :].unsqueeze(1)

        pred = self.forward(image)

        if self.cfg.Model.arch == 'segformer':
            logits = pred.logits
            up_logits = F.interpolate(logits, size=image.shape[-2:],
                                      mode="bicubic", align_corners=False)
            # pred_seg = upsampled_logits.argmax(dim=1)[0]
            
            loss = self.lossfun(up_logits, label)
        elif self.cfg.Model.arch == 'upernet':
            loss = self.lossfun(pred.logits, label)
        else:
            loss = self.lossfun(pred, label)

        output = {"loss": loss}
        self.training_step_outputs.append(output)
        return loss
        
    def on_train_epoch_end(self):
        # print the results
        outputs_gather = self.all_gather(self.training_step_outputs)

        if self.trainer.is_global_zero:
            epoch = int(self.current_epoch)
            train_loss = torch.stack([o['loss']
                                      for o in outputs_gather]).mean().detach()

            print('\n Mean:')
            s = f'  Train:\n'
            s += f'    loss: {train_loss.item():.3f}'
            print(s)
            
            if self.valid_flag:
                s = '  Valid:\n'
                s += f'    loss: {self.valid_loss:.3f}\n'
                for metric in self.valid_metrics:
                    mean_metric_all = self.valid_metrics_mean[metric]
                    s += f'    {metric}: {mean_metric_all:.3f}'
                print(s)                
                self.valid_flag = False

            # log
            d = dict()
            d['epoch'] = epoch
            d['train_loss'] = train_loss

            self.log_dict(d, prog_bar=False, rank_zero_only=True)

            # free up the memory
            self.training_step_outputs.clear()

    def on_validation_start(self):
        # clear GPU cache before the validation
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda() 
     
    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        label = batch["label"]
#        path = batch["filepath"]
       
        if self.cfg.Data.dataset.sampling_type == "2.5d_random":
            # add dummy slices
            offset = self.cfg.Data.dataset.num_slices // 2
            add_img = image[..., 0].unsqueeze(-1)
            for z in range(offset):
                image = torch.cat((add_img, image), dim=-1)
            add_img = image[..., -1].unsqueeze(-1)
            for z in range(offset):
                image = torch.cat((image, add_img), dim=-1)
            
            # permute
            image = image.squeeze(1)    # channel
            image = image.permute(0, 3, 1, 2)    # BDWH
                        
            preds = []
            depth = image.shape[1]
            num_slices = self.cfg.Data.dataset.num_slices

            # sliding in depth direction       
            for d in range(depth - num_slices + 1):
                img = image[:, d:d+num_slices, ...]
                pred = self.forward(img)
                if self.cfg.Model.arch == 'segformer':
                    pred = F.interpolate(pred.logits, size=image.shape[2:4],
                                         mode="bicubic", align_corners=False)
                    preds.append(pred.unsqueeze(-1))
                elif self.cfg.Model.arch == 'upernet':
                    preds.append(pred.logits.unsqueeze(-1))                    
                else:
                    preds.append(pred.unsqueeze(-1))
            
            # concatenate
            pred = torch.cat(preds, dim=-1)
        else:
            pred = self.inferer(image, self.forward)

        # loss
        loss = self.lossfun_valid(pred, label)

        # metrics
        pred = self.post_pred(pred)
            
        for metric in self.valid_metrics:
            self.valid_metrics_fun[metric](y_pred=pred, y=label)
                
        output = {'loss': loss.detach()}
        self.validation_step_outputs.append(output)
        
        return
   
    def on_validation_epoch_end(self):
        epoch = int(self.current_epoch)
        valid_loss = torch.stack([o['loss']
                                  for o in self.validation_step_outputs]).mean().item()
        self.valid_loss = valid_loss

        # log
        d = dict()
        d['epoch'] = epoch
        d['valid_loss'] = valid_loss
        self.valid_metrics_mean = dict()
        for metric in self.valid_metrics:
            # aggregate the final mean dice result
            val = self.valid_metrics_fun[metric].aggregate().item()
            self.valid_metrics_mean[metric] = val
            d[f'valid_{metric}'] = val
            self.valid_metrics_fun[metric].reset()
            
        self.log_dict(d, prog_bar=False, rank_zero_only=True)

        # set flag
        self.valid_flag = True

        # free up the memory
        self.validation_step_outputs.clear()
        
    def move_to(self, obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = self.move_to(v, device)
            return res
        elif isinstance(obj, list):
            res = []
            for v in obj:
                res.append(self.move_to(v, device))
            return res
        elif isinstance(obj, str):
            return obj
        else:
            print('obj (error):', obj)
            raise TypeError("Invalid type for move_to")

    def predict_step(self, batch, batch_idx):
        image = batch["image"]
        path = batch["filepath"]
                   
        if self.cfg.Data.dataset.sampling_type == "2.5d_random":
            # add dummy slices
            offset = self.cfg.Data.dataset.num_slices // 2
            add_img = image[..., 0].unsqueeze(-1)
            for z in range(offset):
                image = torch.cat((add_img, image), dim=-1)
            add_img = image[..., -1].unsqueeze(-1)
            for z in range(offset):
                image = torch.cat((image, add_img), dim=-1)
            
            # permute
            image = image.squeeze(1)    # channel
            image = image.permute(0, 3, 1, 2)    # BDWH
                        
            preds = []
            depth = image.shape[1]
            num_slices = self.cfg.Data.dataset.num_slices

            # sliding in depth direction       
            for d in range(depth - num_slices + 1):
                img = image[:, d:d+num_slices, ...]
                pred = self.forward(img)
                preds.append(pred.unsqueeze(-1))
            
            # concatenate
            pred = torch.cat(preds, dim=-1)
            
            # store in the batch
            batch["pred"] = pred
        else:
            pred = self.inferer(image, self.forward)
        
        batch = [self.post_transforms(b) for b in decollate_batch(batch)]

        assert len(batch) == 1, f"batch size should be 1, but is {len(batch)}."
        
        return

    def on_predict_end(self):
        pass

    def configure_optimizers(self):
        conf_optim = self.cfg.Optimizer

        if hasattr(conf_optim.optimizer, 'params'):
            optimizer_cls, scheduler_cls = get_optimizer(conf_optim)
            optimizer = optimizer_cls(self.parameters(),
                                      **conf_optim.optimizer.params)
        else:
            optimizer_cls, scheduler_cls = get_optimizer(conf_optim)
            optimizer = optimizer_cls(self.parameters())

        if scheduler_cls is None:
            return [optimizer]
        else:
            scheduler = scheduler_cls(
                optimizer, **conf_optim.lr_scheduler.params)
            
            return [optimizer], [scheduler]
               
    def get_progress_bar_dict(self):
        items = dict()

        return items

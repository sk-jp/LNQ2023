import cc3d
import nrrd
import numpy as np
import os

import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from monai.inferers import SliceInferer, SlidingWindowInferer, SlidingWindowSplitter
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric, SurfaceDistanceMetric
from monai.transforms import ResizeWithPadOrCrop
from monai.transforms import Resize

from fix_model_state_dict import fix_model_state_dict
from get_transform import get_transform


def post_process_cc(pred, threshold):
    # get connected components
    cc_map = cc3d.connected_components(pred)

    # check the connected components
    pred_post = np.zeros_like(pred, dtype=np.uint8)
    num_cc = cc_map.max()
#    TH2 = 1000
    max_num_voxel = -1
    argmax_num_voxel = -1
    cc_flag = False
    num_active_cc = 0
    for idx in range(1, num_cc + 1):
        num_voxel = np.sum(cc_map == idx)
        if num_voxel >= threshold:
            pred_post[cc_map == idx] = 1
            cc_flag = True
            num_active_cc += 1
        if num_voxel > max_num_voxel:
            max_num_voxel = num_voxel
            argmax_num_voxel = idx

    if cc_flag is False:
        if num_cc == 0:
            pass
        else:
            pred_post[cc_map == argmax_num_voxel] = 1

    return pred_post


def post_process_resize(pred, org_shape):
    # spatial conversion
    resize = Resize((-1, -1, org_shape[2]), mode="bilinear", dtype=None, align_corners=True)
    pred = resize(torch.tensor(pred).unsqueeze(0))
    pred = (pred > 0).astype(torch.uint8)
        
    padorcrop = ResizeWithPadOrCrop((org_shape[0], org_shape[1], -1))
    pred = padorcrop(pred)
        
    pred = pred[0]      # remove channel dimension
    pred = pred.numpy().astype(np.uint8)
    
    return pred
        

class LightningModule(pl.LightningModule):
    def __init__(self, cfg, transforms=None, post_transforms=None):
        super(LightningModule, self).__init__()
        self.cfg = cfg

        self.txt_logger = cfg.txt_logger
        self.transforms = transforms
        self.post_transforms = post_transforms

        if cfg.General.mode == "train":
            from get_loss import get_loss
            self.lossfun = get_loss(cfg.Loss, train=True)
            self.training_step_outputs = []           
            self.valid_flag = False
            
        if cfg.General.mode == "train" or cfg.General.mode == "valid":
            from get_loss import get_loss
            self.lossfun_valid = get_loss(cfg.Loss, train=False)
            self.validation_step_outputs = []
            self.valid_metrics = ['dice', 'assd']

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
        
        self.post_pred = get_transform(self.cfg.Transform.valid_post)
        self.post_label = AsDiscrete(to_onehot=cfg.Model.params.out_channels, dim=1)

        if cfg.General.mode == "train" or cfg.General.mode == "valid":
            # metrics
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

    def on_validation_start(self):
        # clear GPU cache before the validation
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda() 
     
    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        label = batch["label"]
        filepath = batch["filepath"]
       
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
#            image = image.transpose(2, 3).transpose(1, 2)    # BDWH
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
        elif self.cfg.Model.arch == 'segformer':
            output = self.inferer(image, self.forward)
            logits = output['logits']       # BxCxWxHxD

            preds = []
            for z in range(logits.shape[-1]):
                logit = logits[..., z]  # BxCxWxH
                pred = F.interpolate(logit, size=image.shape[2:4],
                                     mode="bicubic", align_corners=False)
                preds.append(pred.unsqueeze(-1))
            pred = torch.cat(preds, dim=-1)
        else:
            pred = self.inferer(image, self.forward)

        # post
        pred = torch.softmax(pred, dim=1)
        THRESH = 0.5
        pred = torch.where(pred[:, 1, ...] >= THRESH, 1, 0)

        # connected component
        pred = post_process_cc(pred[0].cpu().numpy(), self.cfg.Postprocess.cc_thresh)

        # resize
        org_shape = batch['image_meta_dict']['spatial_shape'][0].cpu().numpy()
        if list(pred.shape) != list(org_shape):
            pred = post_process_resize(pred, org_shape)

        pred_copy = pred.copy()

        # post process for metrics
#        pred = torch.tensor(pred).unsqueeze(0).unsqueeze(0)
        pred = torch.tensor(pred).unsqueeze(0)
        pred = F.one_hot(pred.long()).to(torch.uint8)
        pred = pred.permute(0, 4, 1, 2, 3)

        # read original label file for metrics
        label_filename = filepath[0].replace('ct', 'seg')
        label, _ = nrrd.read(label_filename)
        label = torch.tensor(label).unsqueeze(0).unsqueeze(0)
        
        # loss
        loss = torch.tensor([0.0])    # dummy

        for metric in self.valid_metrics:
            self.valid_metrics_fun[metric](y_pred=pred, y=label)
#            print(f"(post) {metric}: {self.valid_metrics_fun[metric].get_buffer().cpu().numpy()[-1]}")

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
            print('ignore:', obj)

    def on_predict_batch_start(self, batch, batch_idx):
        # clear GPU cache before the prediction
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda() 

    def predict_step(self, batch, batch_idx):
        image = batch["image"]
        filepath = batch["filepath"]
                   
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
#            image = image.transpose(2, 3).transpose(1, 2)    # BDWH
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
        else:
            pred = self.inferer(image, self.forward)

        ## post process
#        pred = torch.argmax(pred, dim=1)
        pred = torch.softmax(pred, dim=1)
        THRESH = 0.5
        pred = torch.where(pred[:, 1, ...] >= THRESH, 1, 0)
        
        # connected component
        pred = post_process_cc(pred[0].cpu().numpy(), self.cfg.Postprocess.cc_thresh)

        # resize        
        org_shape = batch['image_meta_dict']['spatial_shape'][0].cpu().numpy()
        if list(pred.shape) != list(org_shape):
            pred = post_process_resize(pred, org_shape)

        """
        # show
        import SimpleITK as sitk
        img_filename = filepath[0]
        seg_filename = filepath[0].replace('ct', 'seg')
        img = sitk.ReadImage(img_filename)
        img = sitk.GetArrayFromImage(img)
        a_min = -110
        a_max = 140
        img = np.clip((img - a_min) / (a_max - a_min), 0.0, 1.0)
        seg = sitk.ReadImage(seg_filename)
        seg = sitk.GetArrayFromImage(seg)
        pred = pred.transpose(2, 1, 0)    # (z,y,x)
        
        for z in range(img.shape[0]):
            cv2.imshow("img", img[z].T)
            cv2.imshow("seg", seg[z].T * 255)
            cv2.imshow("pred", pred[z].T * 255)
            cv2.waitKey(0)
        """
        
        # save
        import SimpleITK as sitk
        
        org_image = sitk.ReadImage(filepath[0])
        pred = pred.transpose(2, 1, 0)    # (z,y,x)
        output = sitk.GetImageFromArray(pred)
        output.CopyInformation(org_image)
        output = sitk.Cast(output, sitk.sitkUInt8)
        out_filename = os.path.basename(filepath[0]).replace('ct', 'seg')
        out_filepath = f"{self.cfg.General.output_path}/{out_filename}"
        sitk.WriteImage(output, out_filepath)
        
        return

    def on_predict_end(self):
        pass


    def configure_optimizers(self):
        from get_optimizer import get_optimizer

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
            
#            print('opt, sch:', optimizer, scheduler)
            return [optimizer], [scheduler]
        
        
    def get_progress_bar_dict(self):
        items = dict()

        return items

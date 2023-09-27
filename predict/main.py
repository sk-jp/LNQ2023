import argparse
import os
import torch
import torch.multiprocessing

from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch import Trainer, seed_everything

from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureTyped,
    Invertd,
)

import shutil
import warnings

from get_transform import get_transform
from lightning_module import LightningModule
from slices_segment_datamodule import SlicesSegmentDataModule
from read_yaml import read_yaml

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')

torch.cuda.memory._set_allocator_settings("max_split_size_mb:100")


class LitProgressBar(TQDMProgressBar):    
    def init_sanity_tqdm(self):
        bar = super().init_sanity_tqdm()
#        bar.bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} [LAP:{elapsed}/ETA:{remaining}{postfix}]'
        bar.disable = True
        return bar
    
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} [LAP:{elapsed}/ETA:{remaining}{postfix}]'
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
#        bar.bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} [LAP:{elapsed}/ETA:{remaining}{postfix}]'
        bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} [LAP:{elapsed}/ETA:{remaining}{postfix}]'
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        bar.bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} [LAP:{elapsed}/ETA:{remaining}{postfix}]'
        return bar


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--config", required=True, type=str, help="Path to config file")
    arg("--gpus", default="0", type=str, help="GPU IDs")
    arg("--predict_datalist", default=None, type=str, help="Predict datalist")
    arg("--model_pretrained", default=None, type=str, help="Pretrained model")
    arg("--cc_thresh", default=None, type=int, help="threshold in post processing")
    
    return parser


def validate(cfg_name, cfg):
    """ Validation main function
    """
    
    # == initial settings ==
    # random seed
    if isinstance(cfg.General.seed, int):
        seed_everything(seed=cfg.General.seed, workers=True)

    # Patch for "RuntimeError: received 0 items of ancdata" error
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # logger
    cfg.txt_logger = None
    logger = []

    # == callbacks ==
    # progress bar
    progressbar_callback = LitProgressBar()

    # == plugins ==
    plugins = None
    if cfg.General.strategy == "ddp":
        if cfg.Model.arch == 'flexbile_unet':
            strategy = DDPStrategy(find_unused_parameters=True)
        else:
            strategy = DDPStrategy(find_unused_parameters=False)

    # == Trainer ==
    default_root_dir = os.getcwd()
    trainer = Trainer(
        accelerator=cfg.General.accelerator,
        strategy=strategy,
        devices=cfg.General.gpus if cfg.General.accelerator == "gpu" else None,
        num_nodes=cfg.General.num_nodes,
        precision=cfg.General.precision,
        logger=logger,
        callbacks=[progressbar_callback],
        max_epochs=cfg.General.epoch,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        check_val_every_n_epoch=cfg.General.check_val_every_n_epoch,
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.Optimizer.accumulate_grad_batches,
        deterministic=False,
        benchmark=False,
        plugins=plugins,
        default_root_dir=default_root_dir,
    )

    # transforms
    valid_transforms = get_transform(cfg.Transform.valid_3d)
    
    post_transforms = Compose([
        EnsureTyped(keys="image"),
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=valid_transforms,
            orig_keys="image",
#            meta_keys="pred_meta_dict",
#            orig_meta_keys="image_meta_dict",
#            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True),
    ])
    
    # Lightning module and data module
    datamodule = SlicesSegmentDataModule(cfg,
                                         predict_transforms=valid_transforms)
    model = LightningModule(cfg, transforms=valid_transforms,
                            post_transforms=post_transforms)

    # run prediction
    print('*** Start validation***')
    trainer.validate(model, datamodule=datamodule)
    
    return


def predict(cfg_name, cfg):
    """ Predict main function
    """
    
    # == initial settings ==
    # random seed
    if isinstance(cfg.General.seed, int):
        seed_everything(seed=cfg.General.seed, workers=True)

    # Patch for "RuntimeError: received 0 items of ancdata" error
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # logger
    cfg.txt_logger = None
    logger = []

    # == callbacks ==
    # progress bar
    progressbar_callback = LitProgressBar()

    # == plugins ==
    plugins = None
    if cfg.General.strategy == "ddp":
        if cfg.Model.arch == 'flexbile_unet':
            strategy = DDPStrategy(find_unused_parameters=True)
        else:
            strategy = DDPStrategy(find_unused_parameters=False)

    # == Trainer ==
    default_root_dir = os.getcwd()
    trainer = Trainer(
        accelerator=cfg.General.accelerator,
        strategy=strategy,
        devices=cfg.General.gpus if cfg.General.accelerator == "gpu" else None,
        num_nodes=cfg.General.num_nodes,
        precision=cfg.General.precision,
        logger=logger,
        callbacks=[progressbar_callback],
        max_epochs=cfg.General.epoch,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        check_val_every_n_epoch=cfg.General.check_val_every_n_epoch,
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.Optimizer.accumulate_grad_batches,
        deterministic=False,
        benchmark=False,
        plugins=plugins,
        default_root_dir=default_root_dir,
    )

    # transforms
    predict_transforms = get_transform(cfg.Transform.predict_3d)
    
    post_transforms = Compose([
        EnsureTyped(keys="image"),
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=predict_transforms,
            orig_keys="image",
#            meta_keys="pred_meta_dict",
#            orig_meta_keys="image_meta_dict",
#            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True),
    ])
    
    # Lightning module and data module
    datamodule = SlicesSegmentDataModule(cfg,
                                         predict_transforms=predict_transforms)
    model = LightningModule(cfg, transforms=predict_transforms,
                            post_transforms=post_transforms)

    # run prediction
    print('*** Start prediction ***')
    trainer.predict(model, datamodule=datamodule)
    
    return
    

def make_directory(path, remove_dir=False):
    """ Make a directory
        Args:
            path (str): path of the directory to make
            remove_dir (bool): remove the directory if it exists when True
    """    
    if os.path.exists(path):
        if remove_dir is True:
            shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)
    

def main():
    # parse args
    parser = make_parse()
    args = parser.parse_args()
    print('args:', args)

    # Read config
    cfg = read_yaml(fpath=args.config)
    if args.gpus is not None:
        cfg.General.gpus = list(map(int, args.gpus.split(",")))
    if args.predict_datalist:
        cfg.Data.dataset.predict_datalist = args.predict_datalist
    if args.cc_thresh is not None:
        cfg.Postprocess.cc_thresh = args.cc_thresh

    # Make output path and dir
    config_name = os.path.basename(args.config)[:-5]
    print("config:", config_name)
    
#    output_path = Path('./results') / Path(config_name) / Path('/submission_folder')
    output_path = f"./results/{config_name}/submission_folder"
    print('output_path: ', output_path)
    make_directory(output_path, remove_dir=False)
    cfg.General.output_path = output_path
    # Config and Source code backup
#    shutil.copy2(args.config, str(output_path / Path(args.config).name))

    if cfg.General.mode == "valid":
        # Start validation
        validate(cfg_name=config_name, cfg=cfg)
    elif cfg.General.mode == "predict":
        # Start predict
        predict(cfg_name=config_name, cfg=cfg)


if __name__ == '__main__':
    main()

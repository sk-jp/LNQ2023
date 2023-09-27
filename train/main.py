import argparse
from datetime import datetime
import glob
import logging
import os
import torch
import torch.multiprocessing

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch import Trainer, seed_everything

from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureTyped,
    Invertd,
)

from typing import Dict
from pathlib import Path
import shutil
import warnings

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
    arg("--debug", action="store_true", help="Debug mode")
    arg("--predict", action="store_true", help="Run prediction")
    arg("--save_results", action="store_true", help="save prediction results")
    arg("--lr_tune", action="store_true", help="Tune learning rate")
    arg("--lr", default=None, type=float, help="Initial learning rate")
    arg("--seed", default=None, type=int, help="Random seed")
    arg("--rm_cache_dir", action="store_true", help="Remove data cache directory")
    arg("--outdir_ext", default=None, type=str, help="Extention of name of output directory")
    arg("--train_datalist", default=None, type=str, help="Train datalist")
    arg("--valid_datalist", default=None, type=str, help="Valid datalist")
    arg("--predict_datalist", default=None, type=str, help="Predict datalist")
    arg("--model_pretrained", default=None, type=str, help="Pretrained model")
    
    return parser


def train(cfg_name: str, cfg: Dict, output_path: Path) -> None:
    """ Training main function
    """
    
    # == initial settings ==
    # random seed
    if isinstance(cfg.General.seed, int):
        seed_everything(seed=cfg.General.seed, workers=True)

    # Patch for "RuntimeError: received 0 items of ancdata" error
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # debug flag
    debug = cfg.General.debug

    # logger (csv logger, wandb logger, python logging)
    if cfg.General.mode == "predict":
        cfg.txt_logger = None
        logger = []
    else:
        wandb_logger = WandbLogger(
            project=cfg.General.project,
            name=f"{cfg_name}",
            offline=debug)
    
        txt_logger = logging.getLogger("text_logger")
        txt_logger.setLevel(logging.INFO)
        txt_filepath = f'{str(output_path)}/{cfg_name}.log'
        txt_handler = logging.FileHandler(filename=txt_filepath)
        txt_logger.addHandler(txt_handler)
        cfg.txt_logger = txt_logger   
        logger = [wandb_logger]

    # == callbacks ==
    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_path),
        filename=cfg.Model.arch + "-{epoch:02d}-{valid_loss:.2f}",
        monitor='valid_loss',
        save_weights_only=True,
        save_top_k=1,
        mode='min'
    )
    # learning rate monitor
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    # progress bar
    progressbar_callback = LitProgressBar()

    # == plugins ==
    plugins = None
    if not cfg.General.lr_tune:
        if cfg.General.strategy == "ddp":
            if cfg.Model.arch == 'flexbile_unet' \
                or cfg.Model.arch == 'upernet':
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
        callbacks=[checkpoint_callback, lr_monitor_callback, progressbar_callback],
        max_epochs=5 if debug else cfg.General.epoch,
        limit_train_batches=0.05 if debug else 1.0,
        limit_val_batches=1.0 if debug else 1.0,
        check_val_every_n_epoch=cfg.General.check_val_every_n_epoch,
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.Optimizer.accumulate_grad_batches,
        deterministic=False,
        benchmark=False,
        plugins=plugins,
        default_root_dir=default_root_dir,
    )

    # Data module
    datamodule = SlicesSegmentDataModule(cfg)
    
    if cfg.General.mode == "predict":
        # transforms
        predict_transforms = datamodule.predict_transforms
        
        print("predict_transforms:", predict_transforms)

        post_transforms = Compose([
            EnsureTyped(
                keys="pred"
            ),
            Invertd(
                keys="pred",
                transform=predict_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(
                keys="pred",
#                dim=1,
                argmax=True
            ),
        ])
    else:
        post_transforms = None
    
    # Lightning module and data module
    model = LightningModule(cfg, post_transforms=post_transforms)
    
    if cfg.General.lr_tune:
        # Run learning rate finder
        print('*** Start learning rate finder ***')
        lr_finder = trainer.tuner.lr_find(model,
                                          datamodule=datamodule,
                                          min_lr=cfg.Tuner.min_lr,
                                          max_lr=cfg.Tuner.max_lr,
                                          num_training=cfg.Tuner.num_training)
        # suggested LR
        new_lr = lr_finder.suggestion()
        print('Suggested LR:', new_lr)
        
        # remove temporal checkpoint files
        for p in glob.glob(f'{default_root_dir}/lr_find_temp_model_*.ckpt'):
            if os.path.isfile(p):
                os.remove(p)               
    elif cfg.General.mode == "predict":
        # run prediction
        print('*** Start prediction ***')
        trainer.predict(model, datamodule=datamodule)
    else:
        # Start training
        print('*** Start training ***')
        trainer.fit(model, datamodule=datamodule)

def make_directory(path, remove_dir=False):
    """ Make a directory
        Args:
            path (str): path of the directory to make
            remove_dir (bool): remove the directory if it exists when True
    """    
    if os.path.exists(path):
        if remove_dir is True:
            shutil.rmtree(path)
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
    if cfg.General.debug or args.debug:
        cfg.General.debug = True
    else:
        cfg.General.debug = False
    if cfg.General.lr_tune or args.lr_tune:
        cfg.General.lr_tune = True
    else:
        cfg.General.lr_tune = False
    if args.predict:
        cfg.General.mode = "predict"        
    if args.save_results or cfg.General.save_results:
        assert(cfg.General.mode == "predict"), f"mode is {cfg.General.mode}"
        cfg.General.save_results = True        
    if args.lr is not None:
        cfg.Optimizer.optimizer.params.lr = args.lr
    if args.seed is not None:
        cfg.General.seed = args.seed

    if args.train_datalist:
        cfg.Data.dataset.train_datalist = args.train_datalist
    if args.valid_datalist:
        cfg.Data.dataset.valid_datalist = args.valid_datalist
    if args.predict_datalist:
        cfg.Data.dataset.predict_datalist = args.predict_datalist

    if args.model_pretrained:
        cfg.Model.pretrained = args.model_pretrained
        
    # check args and configs  
    assert (cfg.General.mode == "train" or cfg.General.mode == "test" \
        or cfg.General.mode == "predict"), f"cfg.General.mode = {cfg.General.mode}"
    
    if args.rm_cache_dir:
        if os.path.exists(cfg.Data.dataset.cache_dir):
            # remove chche dir
            print(f"Remove the data cache dir: {cfg.Data.dataset.cache_dir}")
            shutil.rmtree(cfg.Data.dataset.cache_dir)

    # Make output path and dir
    path_str = datetime.now().strftime("%y%m%d_%H%M%S_")
    ext_str = ''
    config_name = os.path.basename(args.config)[:-5]
    print("config:", config_name)
    ext_str += f'{config_name}'
    if args.lr is not None:
        ext_str += f'-LR{args.lr}'
    if args.seed is not None:
        ext_str += f'-SEED{args.seed}'
    if cfg.General.mode == "test":
        ext_str += f'-test'
    if cfg.General.mode == "predict":
        ext_str += f'_predict'
    if args.outdir_ext is not None:
        ext_str += f'-{args.outdir_ext}'
    path_str += ext_str
    cfg_name = ext_str
    
    output_path = Path('./results') / Path(config_name) / Path(path_str)
    print('output_path: ', output_path)
    make_directory(output_path, remove_dir=False)
    cfg.General.output_path = output_path
    # Config and Source code backup
    shutil.copy2(args.config, str(output_path / Path(args.config).name))

    # Start train/valid or predict
    train(cfg_name=cfg_name, cfg=cfg, output_path=output_path)


if __name__ == '__main__':
    main()

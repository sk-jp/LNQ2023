"""

IMPORTANT NOTICE:
  Must be change the code in "torch/utils/data/_utils/collate.py" as followings
  
  return elem_type({key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem})
->
  return elem_type({key: collate([d.get(key) for d in batch], collate_fn_map=collate_fn_map) for key in elem})

"""

import csv
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from monai.data import (
#    DataLoader,
    Dataset,
    GridPatchDataset,
    PatchIterd,
    PersistentDataset,
    list_data_collate,
)

from get_transform import get_transform

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


# load csv file
def load_csv(csv_file, topdir, phase="train"):
    rets = []
    with open(csv_file, "r") as f:
        rows = csv.reader(f)
        for row in rows:
#            print("row:", row)
            ct_path = topdir + "/" + row[0]
            if phase == "train" or phase == "valid":
                seg_path = topdir + "/" + row[1]

            # return values
            ret = {}
            ret["image"] = ct_path
            if phase == "train" or phase == "valid":
                ret["label"] = seg_path
            ret["filepath"] = ct_path
            rets.append(ret)
            
    return rets
    

class SlicesSegmentDataModule(pl.LightningDataModule):
    def __init__(self, cfg, predict_transforms=None):
        super(SlicesSegmentDataModule, self).__init__()

        # configs
        self.cfg = cfg
        
        # transforms
        if self.cfg.General.mode == "train":
            self.train_transforms_3d = get_transform(self.cfg.Transform.train_3d)
        elif self.cfg.General.mode == "train" or self.cfg.General.mode == "valid":
            self.valid_transforms_3d = get_transform(self.cfg.Transform.valid_3d)        
        elif self.cfg.General.mode == "predict":
            if predict_transforms is None:
                self.predict_transforms = get_transform(self.cfg.Transform.predict_3d)
            else:
                self.predict_transforms = predict_transforms

    # called once from main process
    # called only within a single process
    def prepare_data(self):
        # prepare data
        pass
    
    # perform on every GPU
    def setup(self, stage):
        assert (self.cfg.Data.dataset.sampling_type == "2d_all" \
            or self.cfg.Data.dataset.sampling_type == "2d_random" \
            or self.cfg.Data.dataset.sampling_type == "2.5d_random" \
            or self.cfg.Data.dataset.sampling_type == "3d")

        if self.cfg.General.mode == "train":
            # datalist
            train_datalist = load_csv(self.cfg.Data.dataset.train_datalist,
                                      self.cfg.Data.dataset.top_dir,
                                      phase="train")
            # dataset 
            if self.cfg.Data.dataset.sampling_type == "2d_all":
                volume_dataset_train = PersistentDataset(
    #            volume_dataset_train = Dataset(
                    data=train_datalist,
                    transform=self.train_transforms_3d,
                    cache_dir=self.cfg.Data.dataset.cache_dir
                )

                # transforms for slices
                train_transforms_2d = get_transform(self.cfg.Transform.train_2d)
                
                # patch fnctionnction
                patch_func = PatchIterd(
                    keys=["image", "label"],
                    patch_size=(None, None, 1),  # dynamic first two dimensions
                    start_pos=(0, 0, 0)
                )

                # dataset for slices
                self.slice_dataset_train = GridPatchDataset(
                    data=volume_dataset_train,
                    patch_iter=patch_func,
                    transform=train_transforms_2d,
                    with_coordinates=False)
            elif self.cfg.Data.dataset.sampling_type == "2d_random" \
                or self.cfg.Data.dataset.sampling_type == "2.5d_random":
                self.slice_dataset_train = PersistentDataset(
    #            self.slice_dataset_train = Dataset(
                    data=train_datalist,
                    transform=self.train_transforms_3d,
                    cache_dir=self.cfg.Data.dataset.cache_dir
                )            
            elif self.cfg.Data.dataset.sampling_type == "3d":
                self.slice_dataset_train = PersistentDataset(
    #            self.slice_dataset_train = Dataset(
                    data=train_datalist,
                    transform=self.train_transforms_3d,
                    cache_dir=self.cfg.Data.dataset.cache_dir
                )            
            
        elif self.cfg.General.mode == "train" or \
            self.cfg.General.mode == "valid":
            # datalist
            valid_datalist = load_csv(self.cfg.Data.dataset.valid_datalist,
                                      self.cfg.Data.dataset.top_dir,
                                      phase="valid")

            # dataset
            self.volume_dataset_valid = PersistentDataset(
#            self.volume_dataset_valid = Dataset(
                data=valid_datalist,
                transform=self.valid_transforms_3d,
                cache_dir=self.cfg.Data.dataset.cache_dir
            )

        elif self.cfg.General.mode == "predict":
            # datalist
            predict_datalist = load_csv(self.cfg.Data.dataset.predict_datalist,
                                        self.cfg.Data.dataset.top_dir,
                                        phase="predict")

            # dataset
            self.volume_dataset_predict = Dataset(
                data=predict_datalist,
                transform=self.predict_transforms,
            )
          
    def train_dataloader(self):
        if self.cfg.Data.dataset.sampling_type == "2d_all":
            train_loader = DataLoader(
                self.slice_dataset_train,
                batch_size=self.cfg.Data.dataloader.batch_size,
                shuffle=self.cfg.Data.dataloader.train.shuffle,
                num_workers=self.cfg.Data.dataloader.num_workers,
                pin_memory=False
            )
        elif self.cfg.Data.dataset.sampling_type == "2d_random" \
            or self.cfg.Data.dataset.sampling_type == "2.5d_random":
            train_loader = DataLoader(
                self.slice_dataset_train,
                batch_size=self.cfg.Data.dataloader.batch_size,
                shuffle=self.cfg.Data.dataloader.train.shuffle,
                num_workers=self.cfg.Data.dataloader.num_workers,
                pin_memory=self.cfg.Data.dataloader.pin_memory,
                persistent_workers=self.cfg.Data.dataloader.persistent_workers,
                collate_fn=list_data_collate
            )
        elif self.cfg.Data.dataset.sampling_type == "3d":
            train_loader = DataLoader(
                self.slice_dataset_train,
                batch_size=self.cfg.Data.dataloader.batch_size,
                shuffle=self.cfg.Data.dataloader.train.shuffle,
                num_workers=self.cfg.Data.dataloader.num_workers,
                pin_memory=False,
                collate_fn=list_data_collate
            )
            
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.volume_dataset_valid,
#            batch_size=self.cfg.Data.dataloader.batch_size,
            batch_size=1,
            shuffle=self.cfg.Data.dataloader.valid.shuffle,
            num_workers=self.cfg.Data.dataloader.num_workers,
            pin_memory=False
        )
        return val_loader

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.volume_dataset_predict,
            batch_size=1,
            shuffle=self.cfg.Data.dataloader.predict.shuffle,
            num_workers=self.cfg.Data.dataloader.num_workers,
            pin_memory=False
        )
        
        return predict_loader


if __name__ == "__main__":
    import cv2
    from read_yaml import read_yaml
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy('file_system')    
    
    cfg = read_yaml(fpath="flexunet_b4_2.5d_random_predict.yaml")
    print('cfg:', cfg)
    
    dm = SlicesSegmentDataModule(cfg)
    dm.prepare_data()

    # splits/transforms
    dm.setup(stage="predict")

    # use data
    for batch in dm.predict_dataloader():
        image = batch["image"]
        path = batch["filepath"]

        print("path:", path)
        print("image:", image.shape)

    """
        for i in range(image.shape[0]):
            cv2.imshow("image", image[i].squeeze().numpy()/4)
            cv2.imshow("label", label[i].squeeze().numpy()/4095)
            cv2.waitKey(0)
    """

    """ 
    for batch in dm.val_dataloader():
        ct = batch["image"]
        seg = batch["label"]
        path = batch["filepath"]

        print("CT:", ct.shape)
        print("Path:", path)
        
        for i in range(ct.shape[4]):
            cv2.imshow("CT", ct[0, 0, :, :, i].numpy())
            cv2.imshow("seg", seg[0, 0, :, :, i].numpy())
            cv2.waitKey(0)
    """

    dm.teardown(stage="predict")

    """
    # lazy load test data
    dm.setup(stage="test")
    for batch in dm.test_dataloader():
        pass

    dm.teardown(stage="test")
    """
import argparse
from pathlib import Path
from typing import Collection, Tuple, Union
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch import Generator

from .base_dataset import BaseDataset

BATCH_SIZE = 128
NUM_WORKERS = 0

class BaseDataModule(pl.LightningDataModule):
    """
    Base DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # Make sure to set the variables below in subclasses
        self.dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.mapping: Collection
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[1] / "data"

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}
        
    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """
    
    def setup(self):
        """
        Split raw data into train, val, test and set dims
        Assign torch.Dataset objects to self.data_train, self.data_val, and self.data_test if needed
        """
    
    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    @staticmethod
    def split_dataset(base_dataset: BaseDataset, fraction: float, seed: int) -> Tuple[BaseDataset, BaseDataset]:
        """
        Split input base_dataset into 2 base datasets, the first of size fraction * size of the base_dataset and the
        other of size (1 - fraction) * size of the base_dataset.
        """
        split_a_size = int(fraction * len(base_dataset))
        split_b_size = len(base_dataset) - split_a_size
        return random_split(  # type: ignore
            base_dataset, [split_a_size, split_b_size], generator=Generator().manual_seed(seed)
        )

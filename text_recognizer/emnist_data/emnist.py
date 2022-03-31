import json
import os
from pathlib import Path
import shutil
import zipfile
import h5py
import toml
import numpy as np
from torchvision import transforms

from .base_dataset import BaseDataset
from .base_data_module import BaseDataModule
from .util import _download_raw_dataset, _sample_to_balance, _augment_emnist_characters

# ESSENTIAL_CHARACTERS = ["<B>", "<S>", "<E>", "<P>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " ", "!", "\"", "#", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "?"]
NUM_SPECIAL_TOKENS = 4
SAMPLE_TO_BALANCE = True  # If true, take at most the mean number of instances per class.
TRAIN_FRAC = 0.8

ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / "emnist_essentials.json"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "byclass.h5"
RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "emnist"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "emnist"

class EMNIST(BaseDataModule):
    """
    "The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19
    and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset."
    From https://www.nist.gov/itl/iad/image-group/emnist-dataset

    The data split we will use is
    EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
    """

    def __init__(self, args=None):
        super().__init__(args)

        if not os.path.exists(ESSENTIALS_FILENAME):
            EMNIST._download_and_process_emnist()
        with open(ESSENTIALS_FILENAME) as f:
            essentials = json.load(f)

        # self.mapping = ESSENTIAL_CHARACTERS
        self.mapping = list(essentials["characters"])
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}

        self.dims = (1, 28, 28,)
        self.output_dims = (1,)

        # https://pytorch.org/vision/0.9/transforms.html
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self, *args, **kwargs) -> None:
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            EMNIST._download_and_process_emnist()
        # with open(ESSENTIALS_FILENAME) as f:
        #     _essentials = json.load(f)

    def setup(self, stage: str = None) -> None:
        # create datasets
        if stage == "fit" or stage is None:
            # h5py is an interface to the HDF5 binary data format
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_trainval = f["x_train"][:]
                self.y_trainval = f["y_train"][:].squeeze().astype(int)

            data_trainval = BaseDataset(self.x_trainval, self.y_trainval, transform=self.transform)
            self.data_train, self.data_val = BaseDataModule.split_dataset(base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=42)

        if stage == "test" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_test = f["x_test"][:]
                self.y_test = f["y_test"][:].squeeze().astype(int)
            self.data_test = BaseDataset(self.x_test, self.y_test, transform=self.transform)

    def __repr__(self):
        basic = f"EMNIST Dataset\nNum classes: {len(self.mapping)}\nMapping: {self.mapping}\nDims: {self.dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data

    @staticmethod
    def _download_and_process_emnist():
        metadata = toml.load(METADATA_FILENAME)
        _download_raw_dataset(metadata, DL_DATA_DIRNAME)

        filename = metadata["filename"]
        dirname = DL_DATA_DIRNAME

        print("Unzipping EMNIST...")
        curdir = os.getcwd()
        os.chdir(dirname)
        zip_file = zipfile.ZipFile(filename, "r")
        zip_file.extract("matlab/emnist-byclass.mat")

        from scipy.io import loadmat  # pylint: disable=import-outside-toplevel

        # NOTE: If importing at the top of module, would need to list scipy as prod dependency.

        print("Loading training data from .mat file")
        data = loadmat("matlab/emnist-byclass.mat")
        x_train = data["dataset"]["train"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
        y_train = data["dataset"]["train"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS
        x_test = data["dataset"]["test"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
        y_test = data["dataset"]["test"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS
        # NOTE that we add NUM_SPECIAL_TOKENS to targets, since these tokens are the first class indices

        if SAMPLE_TO_BALANCE:
            print("Balancing classes to reduce amount of data")
            x_train, y_train = _sample_to_balance(x_train, y_train)
            x_test, y_test = _sample_to_balance(x_test, y_test)

        print("Saving to HDF5 in a compressed format...")
        PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        with h5py.File(PROCESSED_DATA_FILENAME, "w") as f:
            f.create_dataset("x_train", data=x_train, dtype="u1", compression="lzf")
            f.create_dataset("y_train", data=y_train, dtype="u1", compression="lzf")
            f.create_dataset("x_test", data=x_test, dtype="u1", compression="lzf")
            f.create_dataset("y_test", data=y_test, dtype="u1", compression="lzf")

        print("Saving essential dataset parameters to text_recognizer/datasets...")
        mapping = {int(k): chr(v) for k, v in data["dataset"]["mapping"][0, 0]}
        characters = _augment_emnist_characters(list(mapping.values()))
        essentials = {"characters": characters, "input_shape": list(x_train.shape[1:])}
        with open(ESSENTIALS_FILENAME, "w") as f:
            json.dump(essentials, f)

        print("Cleaning up...")
        shutil.rmtree("matlab")
        os.chdir(curdir)


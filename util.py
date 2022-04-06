import logging
import os
from pathlib import Path
from typing import Tuple

import cv2
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset as BaseDataset
from torchvision.transforms import ToTensor

from config import config

logging.basicConfig(level=config.log_level, format="%(message)s")
logger = logging.getLogger()


def imsave_tensor(filename: str, x: Tensor) -> None:
    """Save (C,H,W) tensor."""
    im = x.squeeze().detach().numpy() * 255
    if len(im.shape) == 3:
        im = im.transpose(1, 2, 0)
    elif len(im.shape) != 2:
        raise ValueError
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(Path("output") / filename), im)


def create_label(vi: Tensor, ir: Tensor) -> Tensor:
    ...


class MSRSset(BaseDataset):
    def __init__(self, root: str, train: bool = True) -> None:
        """Initial MSRS dataset."""
        self.root = Path(root)
        self.train = train
        self.vi_train_path = self.root / "Visible" / "train" / "MSRS"
        self.ir_train_path = self.root / "Infrared" / "train" / "MSRS"
        self.vi_test_path = self.root / "Visible" / "test" / "MSRS"
        self.ir_test_path = self.root / "Infrared" / "test" / "MSRS"
        self.train_filenames = os.listdir(self.vi_train_path)
        self.test_filenames = os.listdir(self.vi_test_path)

    def __getitem__(self, key: int) -> Tuple[Tensor, Tensor]:
        if self.train:
            vi_file = self.vi_train_path / self.train_filenames[key]
            ir_file = self.ir_train_path / self.train_filenames[key]
        else:
            vi_file = self.vi_test_path / self.test_filenames[key]
            ir_file = self.ir_test_path / self.test_filenames[key]
        imvi = Image.open(vi_file)
        imir = Image.open(ir_file).convert("L")
        if config.visible_is_gray:
            imvi = imvi.convert("L")
        return self.transform(imvi), self.transform(imir)

    @staticmethod
    def transform(img: Image.Image) -> Tensor:
        return ToTensor()(img)

    def __len__(self) -> int:
        if self.train:
            return len(self.train_filenames)
        else:
            return len(self.test_filenames)

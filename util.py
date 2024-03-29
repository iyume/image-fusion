import itertools
import logging
import os
import warnings
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset as BaseDataset
from torchvision.transforms import ToTensor

from config import config

logging.basicConfig(
    level=(
        config.log_level
        if isinstance(config.log_level, int)
        else config.log_level.upper()
    ),
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger()


def imsave_tensor(filename: str, x: Tensor) -> None:
    """Save (C,H,W) tensor."""
    im = x.squeeze().detach().cpu().numpy() * 255
    if len(im.shape) == 3:
        im = im.transpose(1, 2, 0)
    elif len(im.shape) != 2:
        raise ValueError
    cv2.imwrite(str(Path("output") / filename), im)


class MSRSset(BaseDataset):
    def __init__(self, root: str, train: bool = True) -> None:
        """Initial MSRS dataset."""
        self.name = "MSRS"
        self.root = Path(root)
        self.train = train
        self.vi_train_path = self.root / "Visible" / "train" / "MSRS"
        self.ir_train_path = self.root / "Infrared" / "train" / "MSRS"
        self.label_train_path = self.root / "Label" / "train" / "MSRS"
        self.vi_test_path = self.root / "Visible" / "test" / "MSRS"
        self.ir_test_path = self.root / "Infrared" / "test" / "MSRS"
        self.label_test_path = self.root / "Label" / "test" / "MSRS"
        # self.vi_train_path = self.root / "train" / "vi"
        # self.ir_train_path = self.root / "train" / "ir"
        # self.label_train_path = self.root / "train" / "Segmentation_labels"
        # self.vi_test_path = self.root / "test" / "ir"
        # self.ir_test_path = self.root / "test" / "ir"
        # self.label_test_path = self.root / "test" / "Segmentation_labels"
        self.train_filenames = sorted(os.listdir(self.vi_train_path))
        self.test_filenames = sorted(os.listdir(self.vi_test_path))

    def __getitem__(self, key: int) -> Tuple[Tensor, Tensor, Tensor, bool]:
        if self.train:
            filename = self.train_filenames[key]
            vi_file = self.vi_train_path / filename
            ir_file = self.ir_train_path / filename
            label_file = self.label_train_path / filename
        else:
            filename = self.test_filenames[key]
            vi_file = self.vi_test_path / filename
            ir_file = self.ir_test_path / filename
            label_file = self.label_test_path / filename
        imvi = cv2.imread(str(vi_file))
        imir = cv2.imread(str(ir_file), cv2.IMREAD_GRAYSCALE)
        imlabel = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
        is_night = filename.endswith("N.png")
        imvi = cv2.cvtColor(imvi, cv2.COLOR_BGR2GRAY)  # only for train
        return (
            self.transform(imvi),
            self.transform(imir),
            torch.from_numpy(imlabel).unsqueeze(0),
            is_night,
        )

    @staticmethod
    def transform(img: np.ndarray) -> Tensor:
        return ToTensor()(img)

    def __len__(self) -> int:
        if self.train:
            return len(self.train_filenames)
        else:
            return len(self.test_filenames)


def sobelxy(im: Tensor) -> Tensor:
    """Gradient implement. addWeighted 0.5 and batch suit."""
    kernel = im.new_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    wx = kernel.unsqueeze(0).unsqueeze(0)
    wy = kernel.transpose(1, 0).unsqueeze(0).unsqueeze(0)
    wx = wx.repeat(im.shape[1], 1, 1, 1)
    wy = wy.repeat(im.shape[1], 1, 1, 1)
    im = F.pad(im, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(im, wx, groups=im.shape[1])
    gy = F.conv2d(im, wy, groups=im.shape[1])
    return gx * 0.5 + gy * 0.5  # like addWeighted


def _fspecial_gauss_1d(size: int = 11, sigma: float = 1.5) -> Tensor:
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, size: int = 11, sigma: float = 1.5) -> Tensor:
    """Blur input with 1-D kernel."""
    win = _fspecial_gauss_1d(size, sigma).to(input.device)
    win = win.repeat([input.shape[1]] + [1] * (len(input.shape) - 1))
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)
    C = input.shape[1]
    out = F.pad(input, tuple(itertools.repeat((size - 1) // 2, 4)), mode="reflect")
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(
                out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C
            )
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} "
                f"and win size: {win.shape[-1]}"
            )
    return out

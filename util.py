import logging
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms
from torchvision.transforms import ToTensor

from config import config
from models.bisenet import BiSeNetV2

logging.basicConfig(
    level=(
        config.log_level
        if isinstance(config.log_level, int)
        else config.log_level.upper()
    ),
    format="%(message)s",
)
logger = logging.getLogger()


bisenn = BiSeNetV2(19, aux_mode="pred")
bisenn.load_state_dict(
    torch.load("./models/model_final_v2_city.pth", map_location="cpu"), strict=False
)
bisenn.eval()
bisenn.to(config.device)

bisenn_norm = transforms.Compose(
    [transforms.Normalize(mean=(0.3257, 0.3690, 0.3223), std=(0.2112, 0.2148, 0.2115))]
)

# color map respect to https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
color_semantic = torch.tensor(
    [
        (128, 64, 128),  # 0: road
        (244, 35, 232),  # 1: sidewalk
        (70, 70, 70),  # 2: building
        (102, 102, 156),  # 3: wall
        (190, 153, 153),  # 4: fence
        (153, 153, 153),  # 5: pole
        (250, 170, 30),  # 6: traffic light
        (220, 220, 0),  # 7: traffic sign
        (107, 142, 35),  # 8: vegetation
        (152, 251, 152),  # 9: terrain
        (70, 130, 180),  # 10: sky
        (220, 20, 60),  # 11: person
        (255, 0, 0),  # 12: rider
        (0, 0, 142),  # 13: car
        (0, 0, 70),  # 14: truck
        (0, 60, 100),  # 15: bus
        (0, 80, 100),  # 16: train
        (0, 0, 230),  # 17: motorcycle
        (119, 11, 32),  # 18: bicycle
    ]
)


def imsave_tensor(filename: str, x: Tensor) -> None:
    """Save (C,H,W) tensor."""
    im = x.squeeze().detach().numpy() * 255
    if len(im.shape) == 3:
        im = im.transpose(1, 2, 0)
    elif len(im.shape) != 2:
        raise ValueError
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(Path("output") / filename), im)


def get_semantic_color_map(x: Tensor) -> Tensor:
    """
    Args:
        x: shape (N,C,H,W)

    Return:
        (N,H,W,C)
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if x.shape[1] == 1:
        x = torch.repeat_interleave(x, 3, dim=1)  # requires torch 1.1.0+
    index_map = bisenn(bisenn_norm(x))
    semantic_map = color_semantic[index_map]
    return semantic_map


def create_label(vi: Tensor, ir: Tensor) -> Tensor:
    """
    Args:
        vi: shape (N,C,H,W), C=1
        ir: shape (N,C,H,W), C=1
    """
    ir3c = torch.cat((ir, ir, ir), dim=1)
    semantic_map = bisenn(
        torch.cat(tuple(bisenn_norm(x).unsqueeze(0) for x in ir3c), dim=0)
    )  # (N,H,W) store labels
    label = vi.clone()
    for i in range(semantic_map.shape[0]):  # iter batch size
        mask = (semantic_map[i] == 11) + (semantic_map[i] == 13)
        label[i][:, mask] = ir[i][:, mask]
    return label


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
        if config.visible_is_gray:
            imvi = cv2.cvtColor(imvi, cv2.COLOR_BGR2GRAY)
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


class Sobelxy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        self.convx = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        self.convx.weight.data = kernel.unsqueeze(0).unsqueeze(0)
        self.convy = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        self.convy.weight.data = kernel.T.unsqueeze(0).unsqueeze(0)
        self.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.convx(x) + self.convy(x)


class Scharrxy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32
        )
        self.convx = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        self.convx.weight.data = kernel.unsqueeze(0).unsqueeze(0)
        self.convy = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        self.convy.weight.data = kernel.T.unsqueeze(0).unsqueeze(0)
        self.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.convx(x) + self.convy(x)

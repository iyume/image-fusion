from functools import partial
from pathlib import Path
from typing import OrderedDict, Union

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from config import config
from net import Fusion
from util import MSRSset

testset = MSRSset(config.MSRSdir, train=False)

test_loader = DataLoader(testset)


eval_transform = transforms.Compose(
    [transforms.ToTensor(), partial(torch.unsqueeze, dim=0)]
)


def test() -> None:
    """Test model accurate from test dataset."""


def eval(model: OrderedDict, output_dir: Union[Path, str]) -> None:
    """Evaluate images."""
    net = Fusion()
    net.load_state_dict(model)
    net.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vi_filename = "test_img/vi{:01d}.png"
    ir_filename = "test_img/ir{:01d}.png"
    # if names like 'ir01.png', replace 01d with 02d
    for i in range(10):
        vi = eval_transform(cv2.imread(vi_filename.format(i + 1), cv2.IMREAD_GRAYSCALE))
        ir = eval_transform(cv2.imread(ir_filename.format(i + 1), cv2.IMREAD_GRAYSCALE))
        fused = net(vi, ir)
        cv2.imwrite(
            str(output_dir / f"fusion{i+1}.png"),
            fused.detach().squeeze().numpy() * 255,
        )


if __name__ == "__main__":
    # eval(torch.load("./ckpt/iter_53_of_109.pth"), "result")
    from torchvision.models.segmentation import deeplabv3_resnet50

    vi = eval_transform(cv2.imread("test_img/vi1.png"))
    net = deeplabv3_resnet50(pretrained_backbone=False)
    net.eval()
    output = net(vi)
    ...

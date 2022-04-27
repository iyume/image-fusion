from collections import OrderedDict
from pathlib import Path
from time import time
from typing import Union

import cv2
import torch
from torch import Tensor
from torchvision import transforms

from config import config
from net import AutoEncoder
from util import logger


def run_fusion(vi: Tensor, ir: Tensor) -> Tensor:
    vi_features = net.encode(vi)
    ir_features = net.encode(ir)
    out = net.decode(
        (
            torch.max(vi_features, ir_features),
            torch.max(net.sobelxy(vi_features), net.sobelxy(ir_features)),
        )
    )
    return out


eval_transform = transforms.Compose(
    [transforms.ToTensor(), lambda x: torch.unsqueeze(x, dim=0)]
)

net = AutoEncoder()


def evaluate(state_dict: OrderedDict, output_dir: Union[Path, str]) -> None:
    """Evaluate images."""
    logger.debug(repr(state_dict.keys()))
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    net.to(config.device)
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vi_filename = "test_img/vi{:01d}.png"
    ir_filename = "test_img/ir{:01d}.png"
    with torch.no_grad():
        for i in range(20):
            stime = time()
            vi = eval_transform(
                cv2.imread(vi_filename.format(i + 1), cv2.IMREAD_GRAYSCALE)
            )
            ir = eval_transform(
                cv2.imread(ir_filename.format(i + 1), cv2.IMREAD_GRAYSCALE)
            )
            vi = vi.to(config.device)
            ir = ir.to(config.device)
            out = run_fusion(vi, ir)
            etime = time()
            logger.info(f"test {i+1:02d}: {etime-stime:.4f}")
            cv2.imwrite(
                str(output_dir / f"fusion{i+1:02d}.png"),
                out.detach().cpu().squeeze().numpy() * 255,
            )


if __name__ == "__main__":
    evaluate(
        torch.load("./ckpt/model_MSRS_encoder_final.pth", map_location=config.device),
        "result",
    )

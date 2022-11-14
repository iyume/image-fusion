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
from util import imsave_tensor, logger, sobelxy

torch.set_grad_enabled(False)


def run_fusion(vi: Tensor, ir: Tensor) -> Tensor:
    st = time()
    vi_features = net.encode(vi)
    ir_features = net.encode(ir)
    et = time()
    logger.debug(f"encode vi and ir: {et-st}")

    # vis_vi_features = vi_features.squeeze(0)
    # for i, feature in enumerate(vis_vi_features):
    #     imsave_tensor(f"vis{i}.png", feature)

    st = time()
    vi_grads = sobelxy(vi_features)
    ir_grads = sobelxy(ir_features)
    et = time()
    logger.debug(f"sobelxy vi and ir: {et-st}")

    st = time()
    # features_max = torch.max(vi_features, ir_features)
    # features_gap = features_max - vi_features
    # maybe softmax
    # features = vi_features + features_gap
    out = net.decode(
        (torch.max(vi_features, ir_features), torch.max(vi_grads, ir_grads))
    )
    et = time()
    logger.debug(f"decode: {et-st}")

    return out


eval_transform = transforms.Compose(
    [transforms.ToTensor(), lambda x: torch.unsqueeze(x, dim=0)]
)

net = AutoEncoder()
net.eval()
logger.debug(repr(net))


def evaluate(state_dict: OrderedDict, output_dir: Union[Path, str]) -> None:
    """Evaluate images."""
    net.load_state_dict(state_dict, strict=True)
    net.to(config.device)
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_image = 25
    time_log = []
    for i in range(2, 3):
        st = time()
        vi = eval_transform(
            cv2.imread(f"test_img/vi{i+1:01d}.png", cv2.IMREAD_GRAYSCALE)
        )
        ir = eval_transform(
            cv2.imread(f"test_img/ir{i+1:01d}.png", cv2.IMREAD_GRAYSCALE)
        )
        vi = vi.to(config.device)
        ir = ir.to(config.device)
        out = run_fusion(vi, ir)
        out = out.detach().cpu().squeeze().numpy() * 255
        et = time()
        time_log.append(et - st)
        logger.info(f"test {i+1:02d}: {et-st:.4f}")
        cv2.imwrite(str(output_dir / f"fusion{i+1:02d}.png"), out)
    logger.info(f"sum: {sum(time_log)}  avg: {sum(time_log)/num_image}")


if __name__ == "__main__":
    state_dict = torch.load(
        "./ckpt/revision7/model_MSRS_epoch19.pth", map_location=config.device
    )
    state_dict.pop("sobelxy.convx.weight")  # historical problem
    state_dict.pop("sobelxy.convy.weight")
    evaluate(state_dict, "result")

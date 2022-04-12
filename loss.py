from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from config import config
from util import imsave_tensor


def sobelxy(im: Tensor) -> Tensor:
    kernel = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=config.device
    )
    wx = kernel.unsqueeze(0).unsqueeze(0)
    wy = kernel.transpose(1, 0).unsqueeze(0).unsqueeze(0)
    return F.conv2d(im, wx, padding=1) + F.conv2d(im, wy, padding=1)


def l1_loss(
    vi: Tensor, ir: Tensor, label: Tensor, fused: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    label[:, 0, 0, 0] = 2  # avoid nan
    mask = label == 2
    maxlabel = vi.clone()
    maxlabel[mask] = torch.max(vi, ir)[mask]
    loss_max = F.mse_loss(fused, maxlabel)
    vi_grad = sobelxy(vi)
    ir_grad = sobelxy(ir)
    fused_grad = sobelxy(fused)
    loss_grad = F.mse_loss(fused_grad, torch.max(vi_grad, ir_grad))
    loss_total = loss_max + 10 * loss_grad
    return loss_total, loss_max, loss_grad

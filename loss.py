from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from config import config


def sobelxy(im: Tensor) -> Tensor:
    kernel = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=config.device
    )
    wx = kernel.unsqueeze(0).unsqueeze(0)
    wy = kernel.transpose(1, 0).unsqueeze(0).unsqueeze(0)
    return F.conv2d(im, wx, padding=1) + F.conv2d(im, wy, padding=1)


def l1_loss(vi: Tensor, ir: Tensor, fused: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    loss_in = F.l1_loss(torch.max(vi, ir), fused)
    vi_grad = sobelxy(vi)
    ir_grad = sobelxy(ir)
    fused_grad = sobelxy(fused)
    loss_grad = F.l1_loss(torch.max(vi_grad, ir_grad), fused_grad)
    loss_total = loss_in + 10 * loss_grad
    return loss_total, loss_in, loss_grad

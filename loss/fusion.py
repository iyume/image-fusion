import torch.nn.functional as F
from torch import Tensor


def sobelxy(im: Tensor) -> Tensor:
    kernel = im.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    wx = kernel.unsqueeze(0).unsqueeze(0)
    wy = kernel.transpose(1, 0).unsqueeze(0).unsqueeze(0)
    return F.conv2d(im, wx, padding=1) + F.conv2d(im, wy, padding=1)


def grad_loss(x: Tensor, y: Tensor) -> Tensor:
    return F.mse_loss(sobelxy(x), sobelxy(y))

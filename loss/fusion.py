import torch.nn.functional as F
from torch import Tensor
from util import sobelxy


def grad_loss(x: Tensor, y: Tensor) -> Tensor:
    return F.mse_loss(sobelxy(x), sobelxy(y))

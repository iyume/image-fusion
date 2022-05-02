import math
from typing import Callable, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from util import sobelxy

Conv_T = Callable[[Tensor], Tensor]
Conv2_T = Callable[[Tensor, Tensor], Tensor]


def conv3x3(
    in_channels: int, out_channels: int, bias: bool = True, groups: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        3,
        padding=1,
        bias=bias,
        padding_mode="reflect",
        groups=groups,
    )


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, 1, groups=math.gcd(in_channels, mid_channels)
            ),
            # nn.BatchNorm2d(mid_channels),  # bad layer
            nn.ReLU(True),
            conv3x3(mid_channels, mid_channels),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
        )
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 1)  # never groups this
            if in_channels != out_channels
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        residual = self.conv(x)
        identity = identity + residual
        identity = F.relu_(identity)
        return identity


class AutoEncoder(nn.Module):
    """Implement Bottleneck encoder with groups parameter."""

    def __init__(self) -> None:
        super().__init__()
        groups = 12
        cgrow = (groups, groups * 2, groups * 3)
        cmid = (8, 12)
        self.layer = nn.Sequential(
            conv3x3(1, cgrow[0]),
            Bottleneck(cgrow[0], cgrow[1], cmid[0]),
            Bottleneck(cgrow[1], cgrow[2], cmid[1]),
        )
        # maybe upsample before sobelxy?
        self.sobelxy = sobelxy
        self.upsample = nn.Sequential(
            nn.Conv2d(cgrow[2] * 2, cgrow[2], 1),
            nn.ReLU(True),
        )
        # Bottleneck is not suit for decoder
        self.rlayer = nn.Sequential(
            conv3x3(cgrow[2], cgrow[1], groups=groups),
            nn.ReLU(True),
            conv3x3(cgrow[1], cgrow[0], groups=groups // 2),
            nn.ReLU(True),
            conv3x3(cgrow[0], 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        en = self.encode(x)
        de = self.decode((en, self.sobelxy(en)))
        return de

    def encode(self, x: Tensor) -> Tensor:
        out = self.layer(x)
        return out

    def decode(self, x: Union[Tensor, Tuple[Tensor, Tensor]]) -> Tensor:
        if isinstance(x, tuple):
            x = self.upsample(torch.cat(x, 1))
        out = self.rlayer(x)
        return out

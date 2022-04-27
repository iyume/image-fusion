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


# class Sobelxy(nn.Module):
#     """This module performs slower than `util.sobelxy`."""

#     def __init__(self, channels: int) -> None:
#         super().__init__()
#         kernel = torch.tensor(
#             [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
#         ).reshape(1, 1, 3, 3)
#         self.convx = conv3x3(channels, channels, bias=False, groups=channels)
#         self.convx.weight = nn.parameter.Parameter(kernel.repeat(channels, 1, 1, 1))
#         self.convy = conv3x3(channels, channels, bias=False, groups=channels)
#         self.convy.weight = nn.parameter.Parameter(
#             kernel.transpose(3, 2).repeat(channels, 1, 1, 1)
#         )
#         self.requires_grad_(False)

#     def __call__(self, x: Tensor) -> Tensor:
#         with torch.no_grad():  # ensure no grad
#             gx = F.relu(self.convx(x))
#             gy = F.relu(self.convy(x))
#             return 0.5 * gx + 0.5 * gy


class Bottleneck(nn.Module):
    """Grouped downsample Bottleneck implement."""

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
        identity += residual
        identity = F.relu_(identity)
        return identity


class AutoEncoder(nn.Module):
    """Implement Bottleneck encoder with groups decoder."""

    def __init__(self) -> None:
        super().__init__()
        groups = 12  # multiple of 4
        cgrow = (groups, groups * 2, groups * 3)
        cmid = (8, 12)
        self.layer = nn.Sequential(
            conv3x3(1, cgrow[0]),
            Bottleneck(cgrow[0], cgrow[1], cmid[0]),
            Bottleneck(cgrow[1], cgrow[2], cmid[1]),
        )
        self.sobelxy = sobelxy
        self.downsample = nn.Sequential(
            nn.Conv2d(cgrow[2] * 2, cgrow[2], 1),
            nn.ReLU(True),
        )
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
            x = self.downsample(torch.cat(x, 1))
        out = self.rlayer(x)
        return out

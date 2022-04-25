from typing import Callable, Optional

import torch.nn.functional as F
from torch import Tensor, nn

Conv_T = Callable[[Tensor], Tensor]


def conv3x3(in_channels: int, out_channels: int, groups: int = 1) -> nn.Module:
    return nn.Conv2d(
        in_channels, out_channels, 3, padding=1, padding_mode="reflect", groups=groups
    )


class Bottleneck(nn.Module):
    """Grouped downsample Bottleneck implement."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            # nn.BatchNorm2d(mid_channels),  # must remove
            nn.ReLU(True),
            conv3x3(mid_channels, mid_channels),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
        )
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 1)
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
        self._middle_layer_hook: Optional[Conv_T] = None
        groups = 16
        cgrow = (groups, groups * 2, groups * 3)
        cmid = (8, 12)
        self.encode = nn.Sequential(
            conv3x3(1, cgrow[0]),
            Bottleneck(cgrow[0], cgrow[1], cmid[0]),
            Bottleneck(cgrow[1], cgrow[2], cmid[1]),
        )
        self.decode = nn.Sequential(
            conv3x3(cgrow[2], cgrow[1], groups=groups),
            nn.ReLU(True),
            conv3x3(cgrow[1], cgrow[0], groups=groups),
            nn.ReLU(True),
            conv3x3(cgrow[0], 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        en = self.encode(x)
        if self._middle_layer_hook is not None:
            en = self._middle_layer_hook(en)
        de = self.decode(en)
        return de

    def on_middle_layer(self, func: Conv_T) -> None:
        """Register hook before decoding."""
        self._middle_layer_hook = func

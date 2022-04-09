from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        stride = 1 if in_channels == out_channels or dilation > 1 else 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                stride=stride,
                padding=dilation,
                padding_mode="reflect",
                dilation=dilation,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1, stride=stride, dilation=dilation
                ),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        if self.downsample:
            x = self.downsample(x)
        out += x
        F.relu_(out)
        return out


# class FeatureExtractor(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv0 = nn.Sequential(
#             nn.Conv2d(1, 16, 3, padding=1, bias=False, padding_mode="reflect"),
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#         )
#         self.layer1 = ResBlock(16, 16)
#         self.layer2 = ResBlock(16, 32, dilation=2)

#     def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
#         # feat = torch.cat((vi, ir), dim=1)  # (N,4,H,W)
#         # atten = torch.mean(feat, dim=(2, 3), keepdim=True)  # (N,4,1,1)
#         # atten.sigmoid_()
#         # feat_atten = feat * atten
#         x = self.conv0(x)
#         x1 = self.layer1(x)
#         x2 = self.layer2(x1)
#         return x1, x2


class Fusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv0vi = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.conv0ir = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.layer1vi = ResBlock(16, 32, dilation=2)
        self.layer1ir = ResBlock(16, 32, dilation=2)
        self.layer2vi = ResBlock(32, 64, dilation=2)
        self.layer2ir = ResBlock(32, 64, dilation=2)
        self.fuse0 = nn.Conv2d(32, 16, 1)
        self.fuse1 = nn.Conv2d(64, 32, 1)
        self.fuse2 = nn.Conv2d(128, 64, 1)
        self.rec = nn.Sequential(
            nn.Conv2d(32 + 16, 32, 1), nn.Conv2d(32, 16, 1), nn.Conv2d(16, 1, 1)
        )

    def forward(self, vi: Tensor, ir: Tensor) -> Tensor:
        """vi(N,1,H,W) ir(N,1,H,W)"""
        vi, ir = self.conv0vi(vi), self.conv0ir(ir)
        fuse0 = self.fuse0(torch.cat((vi, ir), dim=1))
        vi, ir = self.layer1vi(vi), self.layer1ir(ir)
        fuse1 = self.fuse1(torch.cat((vi, ir), dim=1))
        # vi, ir = self.layer2vi(vi), self.layer2ir(ir)
        # fuse2 = self.fuse2(torch.cat((vi, ir), dim=1))
        return self.rec(torch.cat((fuse0, fuse1), dim=1))

    def __call__(self, *args, **kwargs) -> Tensor:
        return super().__call__(*args, **kwargs)

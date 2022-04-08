import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                padding_mode="reflect",
            )
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        # F.dropout2d(x, p=0.5, training=config.training, inplace=True)
        x = self.bn(x)  # accelerate conv weight training
        F.relu_(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                stride=stride,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                stride=stride,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 1)
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


class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                1, 32, 7, stride=2, padding=3, bias=False, padding_mode="reflect"
            ),  # downscale
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.layer1 = nn.Sequential(ResBlock(32, 32), ResBlock(32, 32))
        self.layer2 = nn.Sequential(ResBlock(32, 64), ResBlock(64, 64))
        # self.layer3 = nn.Sequential(ResBlock(64, 128), ResBlock(128, 128))

    def forward(self, x: Tensor) -> Tensor:
        # feat = torch.cat((vi, ir), dim=1)  # (N,4,H,W)
        # atten = torch.mean(feat, dim=(2, 3), keepdim=True)  # (N,4,1,1)
        # atten.sigmoid_()
        # feat_atten = feat * atten
        x = self.conv0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        return x


class Fusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encode_vi = FeatureExtractor()
        self.encode_ir = FeatureExtractor()
        self.fuse = nn.Conv2d(128, 64, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.Conv2d(16, 4, 3, 1, 1),
            nn.Conv2d(4, 1, 3, 1, 1),
        )
        self.upscale = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1)

    def forward(self, vi: Tensor, ir: Tensor) -> Tensor:
        """vi(N,3,H,W) ir(N,1,H,W)"""
        encoded_vi = self.encode_vi(vi)
        encoded_ir = self.encode_ir(ir)
        fused = self.fuse(torch.cat((encoded_vi, encoded_ir), dim=1))
        output = self.upscale(self.upsample(fused), vi.size())
        return output

    def __call__(self, *args, **kwargs) -> Tensor:
        return super().__call__(*args, **kwargs)

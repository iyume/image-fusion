import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.transforms import transforms

from models.bisenet import BiSeNetV2

bisenn = BiSeNetV2(19, aux_mode="pred")

bisenn.load_state_dict(
    torch.load("./models/model_final_v2_city.pth", map_location="cpu"), strict=False
)
bisenn.eval()
bisenn.cpu()


bisenn_norm = transforms.Normalize(
    mean=(0.3257, 0.3690, 0.3223), std=(0.2112, 0.2148, 0.2115)
)

# color map respect to https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
color_semantic = torch.tensor(
    [
        (128, 64, 128),  # 0: road
        (244, 35, 232),  # 1: sidewalk
        (70, 70, 70),  # 2: building
        (102, 102, 156),  # 3: wall
        (190, 153, 153),  # 4: fence
        (153, 153, 153),  # 5: pole
        (250, 170, 30),  # 6: traffic light
        (220, 220, 0),  # 7: traffic sign
        (107, 142, 35),  # 8: vegetation
        (152, 251, 152),  # 9: terrain
        (70, 130, 180),  # 10: sky
        (220, 20, 60),  # 11: person
        (255, 0, 0),  # 12: rider
        (0, 0, 142),  # 13: car
        (0, 0, 70),  # 14: truck
        (0, 60, 100),  # 15: bus
        (0, 80, 100),  # 16: train
        (0, 0, 230),  # 17: motorcycle
        (119, 11, 32),  # 18: bicycle
    ]
)


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


class Sobelxy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        self.convx = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        self.convx.weight.data = kernel.unsqueeze(0).unsqueeze(0)
        self.convy = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        self.convy.weight.data = kernel.T.unsqueeze(0).unsqueeze(0)
        self.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.convx(x) + self.convy(x)


class Scharrxy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32
        )
        self.convx = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        self.convx.weight.data = kernel.unsqueeze(0).unsqueeze(0)
        self.convy = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        self.convy.weight.data = kernel.T.unsqueeze(0).unsqueeze(0)
        self.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.convx(x) + self.convy(x)


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
        # vi = self.conv_vi(vi)
        # ir = self.conv_ir(ir)
        # save_tensor("conv_vi.png", vi)
        # save_tensor("conv_ir.png", ir)
        # # x = x.view(x.size(0), -1)  # flatten
        # return self.conv(vi + ir)
        # ir3c = torch.repeat_interleave(ir, 3, dim=1)
        # semantic_ir: Tensor = bisenn(bisenn_norm(ir3c))  # (N,H,W) store labels
        # fused_im = (vi + ScharrXY()(ir)) / 2
        # for i in range(semantic_ir.shape[0]):  # iter batch size
        #     mask = semantic_ir[i] == 11
        #     vi[i][:, mask] = fused_im[i][:, mask]
        encoded_vi = self.encode_vi(vi)
        encoded_ir = self.encode_ir(ir)
        fused = self.fuse(torch.cat((encoded_vi, encoded_ir), dim=1))
        output = self.upscale(self.upsample(fused), vi.size())
        return output

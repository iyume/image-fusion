import os

import cv2
import torch
from torch import Tensor
from torchvision import transforms

from config import config
from net import AutoEncoder
from util import sobelxy


def imsave(filename: str, x: Tensor, root: str = "visualize") -> None:
    os.makedirs(root, exist_ok=True)
    im = x.detach().cpu().squeeze().numpy() * 255
    if len(im.shape) != 2:
        raise RuntimeError
    cv2.imwrite(os.path.join(root, filename), im)


transform = transforms.Compose([transforms.ToTensor(), lambda x: x.unsqueeze(0)])

auto_encoder = AutoEncoder()
auto_encoder.load_state_dict(
    torch.load("./ckpt/model_MSRS_encoder_final.pth", map_location=config.device)
)
auto_encoder.eval()


vi_raw = cv2.imread("./test_img/vi1.png")
ir_raw = cv2.imread("./test_img/ir1.png", cv2.IMREAD_GRAYSCALE)
vi_ycrcb = cv2.cvtColor(vi_raw, cv2.COLOR_BGR2YCR_CB)
vi = transform(vi_ycrcb[..., 0])
ir = transform(ir_raw)

# out = net(vi, ir)


vi_grad = sobelxy(vi)
ir_grad = sobelxy(ir)
grad_max = torch.max(vi_grad, ir_grad)

imsave("vi_grad_night.png", vi_grad)

# imsave("fusion_max.png", out)


...

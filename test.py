import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from config import config
from net import Fusion
from util import MSRSset, imsave_tensor

testset = MSRSset(config.MSRSdir, train=False)

test_loader = DataLoader(testset, batch_size=5)

first_batch_vi, first_batch_ir = next(iter(test_loader))

net = Fusion()
net.load_state_dict(torch.load("ckpt/model_final.pth"))
net.eval()

output = net(first_batch_vi, first_batch_ir)

for i, im in enumerate(output):
    imsave_tensor(f"fusion_trained{i}.png", im)

import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from config import config
from net import FeatureFusion, Fusion
from util import MSRSset, imsave_tensor

trainset = MSRSset(config.MSRSdir, train=True)
testset = MSRSset(config.MSRSdir, train=False)

train_loader = DataLoader(trainset, batch_size=20)

# for i, data in enumerate(train_loader):
#     ...


test_vi = Image.open("./MSRS/Visible/train/MSRS/00029N.png")
test_ir = Image.open("./MSRS/Infrared/train/MSRS/00029N.png").convert("L")
# unsqueeze to (1,C,H,W)
test_vi = ToTensor()(test_vi).unsqueeze_(0)
test_ir = ToTensor()(test_ir).unsqueeze_(0)


ffnn = FeatureFusion()
output = ffnn(test_vi, test_ir)

fnn = Fusion()
# output = fnn(test_vi, test_ir)

output = output.squeeze(0)
for i in range(output.shape[0]):
    imsave_tensor(f"output{i}.png", output[i])


...

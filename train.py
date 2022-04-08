from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import config
from net import Fusion
from util import MSRSset, imsave_tensor, logger

trainset = MSRSset(config.MSRSdir, train=True)
testset = MSRSset(config.MSRSdir, train=False)

train_loader = DataLoader(trainset, batch_size=config.batch_size)


def train(ckpt_path: str = "ckpt") -> None:
    net = Fusion()
    optimizer = torch.optim.Adam(net.parameters())
    l1loss = torch.nn.L1Loss()
    for i, batch in enumerate(train_loader):
        vi, ir = batch
        optimizer.zero_grad(set_to_none=True)
        logger.info(f"# starting {i}/{len(train_loader)}")
        fused_im = net(vi, ir)
        loss = l1loss(vi, fused_im)
        loss.backward()
        optimizer.step()
        logger.info(f"loss: {loss}\n")
        if (i + 1) % (len(train_loader) // 4) == 0:
            pthname = Path(ckpt_path) / f"iter_{i}_of_{len(train_loader)}.pth"
            torch.save(net.state_dict(), pthname)
            logger.info(f"model saved at {pthname}")


train()

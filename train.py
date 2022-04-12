from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from config import config
from loss import l1_loss
from net import Fusion
from util import MSRSset, logger

trainset = MSRSset(config.MSRSdir, train=True)
train_loader = DataLoader(trainset, batch_size=config.batch_size)


def _train(net: Fusion) -> None:
    vi: torch.Tensor
    ir: torch.Tensor
    label: torch.Tensor
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    for i, batch in enumerate(train_loader):
        vi, ir, label, _ = batch
        vi = vi.to(device=config.device)
        ir = ir.to(device=config.device)
        optimizer.zero_grad()
        fused_im = net(vi, ir)
        loss_total, loss_max, loss_grad = l1_loss(vi, ir, label, fused_im)
        loss_total.backward()
        optimizer.step()
        logger.info(
            "iter: {}  loss_total: {:.4f}  loss_max: {:.4f}  loss_grad: {:.4f}".format(
                f"{i + 1}/{len(train_loader)}",
                loss_total,
                loss_max,
                loss_grad,
            )
        )


def train(
    epoch: int, init_state_dict: Optional[OrderedDict] = None, ckpt_dir: str = "ckpt"
) -> None:
    previous_checkpoint = None
    for epo in range(epoch):
        net = Fusion()
        checkpoint = Path(ckpt_dir) / "model_{}_{}.pth".format(
            trainset.name, "final" if epo == epoch - 1 else f"epo{epo + 1}"
        )
        if init_state_dict is not None:
            net.load_state_dict(init_state_dict)
            init_state_dict = None
        if epo != 0:
            net.load_state_dict(torch.load(previous_checkpoint))
        previous_checkpoint = checkpoint
        net.to(config.device)
        _train(net)
        torch.save(net.state_dict(), checkpoint)
        logger.info(f"epoch {epo + 1} training complete, model saved at {checkpoint}")


if __name__ == "__main__":
    train(config.epoch)

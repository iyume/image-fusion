from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from config import config
from loss import SSIM, grad_loss
from net import AutoEncoder
from util import MSRSset, logger, sobelxy

trainset = MSRSset(config.MSRSdir, train=True)
train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)

net = AutoEncoder()


@net.on_middle_layer
def _(features: torch.Tensor) -> torch.Tensor:
    return torch.cat((features, sobelxy(features)), 1)


def _train(
    net: AutoEncoder, optimizer: torch.optim.Optimizer, train_state: dict
) -> None:
    batch: Tuple[torch.Tensor, ...]
    pixel_loss = torch.nn.MSELoss()
    ssim = SSIM(1, channel=1)
    loss_total_sum = 0.0
    for i, batch in enumerate(train_loader):
        vi, ir, label, _ = batch
        vi = vi.to(device=config.device)
        ir = ir.to(device=config.device)
        optimizer.zero_grad()
        out = net(vi)
        loss_pixel = pixel_loss(out, vi)
        loss_ssim = -ssim(out, vi) + 1
        loss_grad = grad_loss(out, vi)
        loss_total = loss_pixel + 100 * loss_ssim + 10 * loss_grad
        loss_total.backward()
        optimizer.step()
        logger.info(
            "iter: {}  loss_total: {:.4f}  loss_pixel: {:.4f}  loss_ssim: {:.4f}  loss_grad: {:.4f}".format(
                f"{i + 1}/{len(train_loader)}",
                loss_total,
                loss_pixel,
                loss_ssim,
                loss_grad,
            )
        )  # __format__ magic
        loss_total_sum += loss_total.item()
    train_state.clear()
    train_state.update({"loss_total_avg": f"{loss_total_sum / len(train_loader):.4f}"})


def train(
    num_epoch: int,
    init_state_dict: Optional[OrderedDict] = None,
    ckpt_dir: str = "ckpt",
) -> None:

    net.to(config.device)
    if init_state_dict is not None:
        net.load_state_dict(init_state_dict)
        init_state_dict = None
    logger.debug(repr(net))
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    for epoch in range(num_epoch):
        checkpoint = Path(ckpt_dir) / "model_{}_{}.pth".format(
            trainset.name, f"epoch{epoch + 1}"
        )
        train_state = {}
        _train(net, optimizer, train_state)
        logger.info(f"epoch {epoch + 1} training complete")
        logger.info("  ".join(f"{k}: {v}" for k, v in train_state.items()))
        torch.save(net.state_dict(), checkpoint)
        logger.info(f"model saved at {checkpoint}")


if __name__ == "__main__":
    train(config.epoch)

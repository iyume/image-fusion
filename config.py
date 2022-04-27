import torch


class Config:
    """Training config."""

    log_level = "DEBUG"  # only INFO and DEBUG level support
    MSRSdir = "./MSRS"
    epoch = 3
    learning_rate = 1e-3
    batch_size = 6
    device = torch.device("cpu")


# change it on argparse
config = Config()

import torch


class Config:
    log_level = 20
    MSRSdir = "./MSRS"
    visible_is_gray = True
    training = True
    batch_size = 10
    device = torch.device("cuda:0")


# change it on argparse
config = Config()

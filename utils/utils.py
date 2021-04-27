import os
import random
import torch
import numpy as np


def create_dir(dir_path):
    """
    Creates a directory if not exist.
    """
    os.makedirs(dir_path, exist_ok=True)

def create_dirs(dir_paths):
    """
    Creates a directory for all given paths
    """
    for dir_path in dir_paths:
        os.makedirs(dir_path, exist_ok=True)

def collate_fn(batch):
    return tuple(zip(*batch))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

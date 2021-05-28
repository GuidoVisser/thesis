import os
import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def test_epoch(model: torch.nn.Module, extra_models: list, data_loader: DataLoader) -> dict:
    """
    Test the model on one epoch
    
    Args:
        model (torch.nn.Module)
        extra_models (list): list of models used in the pipeline of training (these will not be trained)
        data_loader (DataLoader)

    Returns:
        output (dict): Dict containing loss values of the epochs
    """
    output = {}

    return output

def train_epoch(model: torch.nn.Module, extra_models: list, train_loader: DataLoader, optimizer:torch.optim.Optimizer) -> dict:
    """
    Train the model for one epoch

    Args:
        model (torch.nn.Module)
        extra_models (list): list of models used in the pipeline of training (these will not be trained)
        data_loader (DataLoader)
        optimizer (torch.optim.Optimizer)

    Returns:
        output (dict): Dict containing loss values of the epochs
    """
    output = {}
    return output

def demo_model(data_dir: str, demo_dir: str, model: torch.nn.Module, extra_models: list, epoch: int) -> None:
    pass
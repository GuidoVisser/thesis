from torch.nn.modules.activation import Sigmoid
import torch

def sigmoid_smoothing(x):
    return 2 * Sigmoid(5 * x) - 1

def composite_rgba(composite: torch.Tensor, rgba: torch.Tensor) -> torch.Tensor:
    """
    Add a new layer to an existing RGBa composite

    composite (torch.Tensor): the current composite
    rgba (torch.Tensor):      the newly added layer
    """

    comp = composite * .5 + .5
    alpha = rgba[:, 3:4] * .5 + .5
    new_layer = rgba * .5 + .5

    return ((1. - alpha) * comp + alpha * new_layer) * 2 - 1
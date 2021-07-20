import torch
import torch.nn as nn
from torch.nn.modules.loss import L1Loss, 

from .compositing import composite
from .utils import sigmoid_smooting

class RGBReconstructionLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = L1Loss()

    def __call__(self, layers, frame, layer_order):
        reconstruction = composite(layers, layer_order)
        return self.l1(reconstruction, frame)
            

class FlowReconstructionLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = L1Loss()

    def __call__(self, layers, gt, layer_order, conf):
        reconstruction = composite(layers, layer_order)
        return conf * self.l1(reconstruction, gt)

class RegularizationLoss(nn.Module):
    
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def __call__(self, layers):
        return layers
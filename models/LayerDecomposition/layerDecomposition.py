import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class LayerDecompositer(nn.Module):
    def __init__(self,
                 dataloader: DataLoader,
                 loss_module: nn.Module,
                 network: nn.Module,
                 learning_rate: float):
        super().__init__()

        self.dataloader = dataloader
        self.loss_module = loss_module
        self.net = network
        self.optimizer = torch.optim.Adam(self.net.parameters(), learning_rate)

    def train(self):

        for _, (input, targets) in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            output = self.net(input)
            loss = self.loss_module(output, targets)
            loss.backward()
            self.optimizer.step()

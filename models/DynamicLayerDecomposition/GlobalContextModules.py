import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet50Backbone(nn.Module):

    def __init__(self):
        super().__init__()
        
        resnet = resnet50(pretrained=True)
        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2   = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

    def forward(self, f):
        x   = self.conv1(f) 
        x   = self.bn1(x)
        x   = self.relu(x)     # 1/2, 64
        x   = self.maxpool(x)  # 1/4, 64
        f4  = self.res2(x)     # 1/4, 256
        f8  = self.layer2(f4)  # 1/8, 512
        f16 = self.layer3(f8)  # 1/16, 1024

        return f16#, f8, f4


class KeyValueEncoder(nn.Module):
    def __init__(self, indim, keydim, valdim) -> None:
        super().__init__()

        self.backbone = ResNet50Backbone()

        self.key_layer = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)
        self.val_layer = nn.Conv2d(indim, valdim, kernel_size=3, padding=1)


    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x)

        key = self.key_layer(x)
        val = self.val_layer(x)
        
        return key, val


class GlobalContextVolume(object):
    def __init__(self, context_volume: torch.Tensor) -> None:
        super().__init__()

        self.context_volume = context_volume

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Returns a context distribution defined by the global context and the local query

        D_t = q(x_t) * G

        Args:
            query (torch.Tensor[B x C_N x H, W])

        Returns:
            context_dist (torch.Tensor[B x C_m x H x W])

        """
        pass

    def update(self, local_contexts: torch.Tensor) -> None:
        """
        Update the global context volume using the local context matrices at different timesteps
        v1:
            the average of the local context matrices is taken
        
        Args:
            local_contexts (list[ torch.Tensor[B x C_m x C_n] ] -- length=T)
        """
        self.context_volume = torch.mean(torch.stack(local_contexts, dim=0), dim=0, keepdim=False)
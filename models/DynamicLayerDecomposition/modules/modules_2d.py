import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.DynamicLayerDecomposition.modules.base_modules import ConvBlock
from models.DynamicLayerDecomposition.modules.base_modules import GlobalContextVolume
from models.DynamicLayerDecomposition.modules.base_modules import MemoryEncoder
from models.DynamicLayerDecomposition.modules.base_modules import KeyValueEncoder


class ConvBlock2D(ConvBlock):
    """
    ConvBlock with 2D convolutions/transposed convolutions
    """
    def __init__(self, in_channels, out_channels, ksize=4, stride=1, dil=1, norm=None, activation='relu', transposed=False):
        super().__init__(out_channels, ksize, stride, dil, norm, activation)

        if transposed:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, ksize, stride=stride, dilation=dil)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride=stride, dilation=dil)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Compute necessary padding and cropping because pytorch doesn't have pad=same."""
        height, width = x.shape[-2:]
        if isinstance(self.conv, nn.modules.ConvTranspose2d):
            desired_height = height * self.s
            desired_width = width * self.s
            pady = 0
            padx = 0
        else:
            desired_height = height // self.s
            desired_width = width // self.s
            pady = .5 * (self.s * (desired_height - 1) + (self.k - 1) * (self.d - 1) + self.k - height)
            padx = .5 * (self.s * (desired_width - 1) + (self.k - 1) * (self.d - 1) + self.k - width)
        x = F.pad(x, [int(np.floor(padx)), int(np.ceil(padx)), int(np.floor(pady)), int(np.ceil(pady))])
        x = self.conv(x)

        if x.shape[-2] != desired_height or x.shape[-1] != desired_width:
            cropy = x.shape[-2] - desired_height
            cropx = x.shape[-1] - desired_width
            x = x[:, :, int(np.floor(cropy / 2.)):-int(np.ceil(cropy / 2.)),
                int(np.floor(cropx / 2.)):-int(np.ceil(cropx / 2.))]
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class GlobalContextVolume2D(GlobalContextVolume):
    """
    Global Context Volume usable with 2D convolutions
    """
    def __init__(self, keydim: int, valdim: int, topk: int) -> None:
        super().__init__(keydim, valdim, topk)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Returns a context distribution defined by the global context and the local query

        D_t = q(x_t) * G

        Args:
            query (torch.Tensor[B x C_k x H, W])

        Returns:
            context_dist (torch.Tensor[B x C_v x H x W])

        """
        B, _, H, W = query.shape

        query = query.view(B, -1, H*W)                          # -> [B x C_k x HW]
        context_dist = torch.matmul(self.context_volume, query) # -> [B x C_v x HW]
        context_dist = context_dist.view(B, -1, H, W)           # -> [B x C_v x H x W]

        if self.topk is not None:
            context_dist = torch.topk(context_dist, self.topk, dim=1)

        return context_dist


class MemoryEncoder2D(MemoryEncoder):
    """
    Memory Encoder usable with 2D convolutions
    """
    def __init__(self, conv_channels: int, keydim: int, valdim: int, topk: int, backbone: nn.Module, gcv: GlobalContextVolume) -> None:
        super().__init__(keydim, valdim)

        self.memory_encoder = KeyValueEncoder(nn.Conv2d, conv_channels, keydim, valdim, backbone)
        self.global_context = gcv(keydim, valdim, topk)

    def _get_input(self, spatiotemporal_noise: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """
        Get the input for a single timestep to add to the global context

        Args:
            spatiotemoral_noise (torch.Tensor): noise tensor of the entire video
            frame_idx (int): index of first frame in the input
        """
        return spatiotemporal_noise[:, :, frame_idx]

    def _get_frame_idx_iterator(self, length_video: int):
        """
        Get a range object that specifies which frame indices should be considered in the memory

        Args:
            length_video (int): number of frames in the video
        """
        return range(length_video)

    def _get_context_from_key_value_pair(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Takes as argument a key value pair and returns the corresponding context matrix

        The context is calculated according to:
            C [B x C_v x C_k] = v [B x C_v x HW] @ k [B x HW x C_k] 

        Args:
            key   (torch.Tensor[B x C_k x H x W])
            value (torch.Tensor[B x C_v x H x W])

        Returns:
            context (torch.Tensor[B x C_v x C_k])
        """
        B, _, H, W = key.shape
        
        key   = key.view(B, -1, H*W).permute(0, 2, 1)
        value = value.view(B, -1, H*W)

        context = torch.matmul(value, key)

        return context

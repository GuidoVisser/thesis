import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.DynamicLayerDecomposition.modules.base_modules import ConvBlock
from models.DynamicLayerDecomposition.modules.base_modules import GlobalContextVolume
from models.DynamicLayerDecomposition.modules.base_modules import MemoryEncoder

class ConvBlock3D(ConvBlock):
    """
    ConvBlock with 3D convolutions/transposed convolutions
    """
    def __init__(self, in_channels, out_channels, ksize=(4, 4, 4), stride=(1, 1, 1), dil=(1, 1, 1), norm=None, activation='relu', transposed=False):
        super().__init__(out_channels, ksize, stride, dil, norm, activation)

        if transposed:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, ksize, stride=stride, dilation=dil)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, ksize, stride=stride, dilation=dil)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Compute necessary padding and cropping because pytorch doesn't have pad=same."""
        time_steps, height, width = x.shape[-3:]
        if isinstance(self.conv, nn.modules.ConvTranspose3d):
            desired_length = time_steps * self.s[0] 
            desired_height = height * self.s[1]
            desired_width  = width  * self.s[2]
            padt = 0
            pady = 0
            padx = 0
        else:
            desired_length = time_steps // self.s[0]
            desired_height = height     // self.s[1]
            desired_width  = width      // self.s[2]
            padt = .5 * (self.s[0] * (desired_length - 1) + (self.k[0] - 1) * (self.d[0] - 1) + self.k[0] - time_steps)
            pady = .5 * (self.s[1] * (desired_height - 1) + (self.k[1] - 1) * (self.d[1] - 1) + self.k[1] - height)
            padx = .5 * (self.s[2] * (desired_width  - 1) + (self.k[2] - 1) * (self.d[2] - 1) + self.k[2] - width)
        x = F.pad(x, [int(np.floor(padx)), int(np.ceil(padx)), int(np.floor(pady)), int(np.ceil(pady)), int(np.floor(padt)), int(np.ceil(padt))])
        x = self.conv(x)
        if x.shape[-3] != desired_length or x.shape[-2] != desired_height or x.shape[-1] != desired_width:
            cropt = x.shape[-3] - desired_length
            cropy = x.shape[-2] - desired_height
            cropx = x.shape[-1] - desired_width
            x = x[:, :, int(np.floor(cropt / 2.)):-int(np.ceil(cropt / 2.)),
                        int(np.floor(cropy / 2.)):-int(np.ceil(cropy / 2.)),
                        int(np.floor(cropx / 2.)):-int(np.ceil(cropx / 2.))]
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class GlobalContextVolume3D(GlobalContextVolume):
    """
    Global Context Volume usable with 3D convolutions
    """
    def __init__(self, keydim: int, valdim: int, topk: int) -> None:
        super().__init__(keydim, valdim, topk)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Returns a context distribution defined by the global context and the local query

        D_t = q(x_t) * G

        Args:
            query (torch.Tensor[B, C_k, T, H, W])

        Returns:
            context_dist (torch.Tensor[B, C_v, T, H, W])

        """
        B, _, T, H, W = query.shape

        query = query.view(B, -1, T*H*W)                        # -> [B, C_k, THW]
        context_dist = torch.matmul(self.context_volume, query) # -> [B, C_v, THW]
        context_dist = context_dist.view(B, -1, T, H, W)        # -> [B, C_v, T, H, W]

        if self.topk is not None:
            context_dist = torch.topk(context_dist, self.topk, dim=1).values

        return context_dist


class MemoryEncoder3D(MemoryEncoder):
    """
    Memory Encoder usable with 3D convolutions
    """
    def __init__(self, conv_channels: int, keydim: int, valdim: int, topk: int, gcv: GlobalContextVolume, mem_freq: int = 4, timesteps: int = 16) -> None:
        super().__init__(keydim, valdim)

        self.key_value_encoder      = KeyValueEncoder(nn.Conv3d, conv_channels, keydim, valdim)
        self.global_context         = gcv(keydim, valdim, topk)

        self.timesteps = timesteps
        self.mem_freq  = mem_freq

    def _get_context_from_key_value_pair(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Takes as argument a key value pair and returns the corresponding context matrix

        The context is calculated according to:
            C [B x C_v x C_k] = v [B x C_v x THW] @ k [B x THW x C_k] 

        Args:
            key   (torch.Tensor[B x C_k x T x H x W])
            value (torch.Tensor[B x C_v x T x H x W])

        Returns:
            context (torch.Tensor[B x C_v x C_k])
        """
        B, _, T, H, W = key.shape
        
        key   = key.view(B, -1, T*H*W).permute(0, 2, 1)
        value = value.view(B, -1, T*H*W)

        context = torch.matmul(value, key)

        return context

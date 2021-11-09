import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.DynamicLayerDecomposition.modules.base_modules import ConvBlock
from models.DynamicLayerDecomposition.modules.base_modules import GlobalContextVolume
from models.DynamicLayerDecomposition.modules.base_modules import MemoryEncoder
from models.DynamicLayerDecomposition.modules.base_modules import MemoryReader
from models.DynamicLayerDecomposition.modules.base_modules import KeyValueEncoder
from models.DynamicLayerDecomposition.modules.base_modules import LayerDecompositionAttentionMemoryNet


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
    def __init__(self, keydim: int, valdim: int) -> None:
        super().__init__(keydim, valdim)

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

        return context_dist


class MemoryEncoder3D(MemoryEncoder):
    """
    Memory Encoder usable with 3D convolutions
    """
    def __init__(self, conv_channels: int, keydim: int, valdim: int, backbone: nn.Module, mem_freq: int = 4, timesteps: int = 16) -> None:
        super().__init__(keydim, valdim)

        self.memory_encoder = KeyValueEncoder(nn.Conv3d, conv_channels, keydim, valdim, backbone)
        self.global_context = GlobalContextVolume3D(keydim, valdim)

        self.timesteps = timesteps
        self.mem_freq  = mem_freq

    def _get_input(self, spatiotemporal_noise: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """
        Get the input for a single timestep to add to the global context

        Args:
            spatiotemoral_noise (torch.Tensor): noise tensor of the entire video
            frame_idx (int): index of first frame in the input
        """
        return spatiotemporal_noise[:, :, frame_idx:frame_idx + self.timesteps]

    def _get_frame_idx_iterator(self, length_video: int):
        """
        Get a range object that specifies which frame indices should be considered in the memory

        Args:
            length_video (int): number of frames in the video
        """
        return range(0, length_video - self.timesteps, self.mem_freq)

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


class LayerDecompositionAttentionMemoryNet3D(LayerDecompositionAttentionMemoryNet):
    """
    Layer Decomposition Attention Memory Net with 3D convolutions
    """
    def __init__(self, conv_channels=64, in_channels=16, valdim=128, keydim=64, max_frames=200, coarseness=10, mem_freq=4, timesteps=16, do_adjustment=True, shared_encoder=True):
        super().__init__(max_frames, coarseness, do_adjustment)

        # initialize foreground encoder and decoder
        query_backbone = nn.ModuleList([
            ConvBlock3D(in_channels,       conv_channels,     ksize=(4, 4, 4), stride=(2, 2, 2)),                                                  # 1/2
            ConvBlock3D(conv_channels,     conv_channels * 2, ksize=(4, 4, 4), stride=(1, 2, 2),                norm=nn.BatchNorm3d, activation='leaky'),  # 1/4
            ConvBlock3D(conv_channels * 2, conv_channels * 4, ksize=(4, 4, 4), stride=(2, 2, 2),                norm=nn.BatchNorm3d, activation='leaky'),  # 1/8
            ConvBlock3D(conv_channels * 4, conv_channels * 4, ksize=(4, 4, 4), stride=(1, 2, 2),                norm=nn.BatchNorm3d, activation='leaky'),  # 1/16
            ConvBlock3D(conv_channels * 4, conv_channels * 4, ksize=(4, 4, 4), stride=(1, 1, 1), dil=(2, 2, 2), norm=nn.BatchNorm3d, activation='leaky'),  # 1/16
            ConvBlock3D(conv_channels * 4, conv_channels * 4, ksize=(4, 4, 4), stride=(1, 1, 1), dil=(2, 2, 2), norm=nn.BatchNorm3d, activation='leaky')]) # 1/16
                
        self.memory_reader = MemoryReader(nn.Conv3d, conv_channels * 4, keydim, valdim, query_backbone)

        if shared_encoder:
            self.memory_encoder = MemoryEncoder3D(conv_channels * 4, keydim, valdim, query_backbone, mem_freq, timesteps)
        else:
            memory_backbone = nn.ModuleList([
                ConvBlock3D(in_channels,       conv_channels,     ksize=(4, 4, 4), stride=(2, 2, 2)),                                                          # 1/2
                ConvBlock3D(conv_channels,     conv_channels * 2, ksize=(4, 4, 4), stride=(2, 2, 2),                norm=nn.BatchNorm3d, activation='leaky'),  # 1/4
                ConvBlock3D(conv_channels * 2, conv_channels * 4, ksize=(4, 4, 4), stride=(2, 2, 2),                norm=nn.BatchNorm3d, activation='leaky'),  # 1/8
                ConvBlock3D(conv_channels * 4, conv_channels * 4, ksize=(4, 4, 4), stride=(2, 2, 2),                norm=nn.BatchNorm3d, activation='leaky'),  # 1/16
                ConvBlock3D(conv_channels * 4, conv_channels * 4, ksize=(4, 4, 4), stride=(1, 1, 1), dil=(2, 2, 2), norm=nn.BatchNorm3d, activation='leaky'),  # 1/16
                ConvBlock3D(conv_channels * 4, conv_channels * 4, ksize=(4, 4, 4), stride=(1, 1, 1), dil=(2, 2, 2), norm=nn.BatchNorm3d, activation='leaky')]) # 1/16

            self.memory_encoder = MemoryEncoder3D(conv_channels * 4, keydim, valdim, memory_backbone, mem_freq, timesteps)

        decoder_in_channels = conv_channels * 4 + valdim * 2

        self.decoder = nn.ModuleList([
            ConvBlock3D(decoder_in_channels,   conv_channels * 4, ksize=(4, 4, 4), stride=(1, 2, 2), norm=nn.BatchNorm3d, transposed=True),  # 1/8
            ConvBlock3D(conv_channels * 2 * 4, conv_channels * 2, ksize=(4, 4, 4), stride=(2, 2, 2), norm=nn.BatchNorm3d, transposed=True),  # 1/4
            ConvBlock3D(conv_channels * 2 * 2, conv_channels,     ksize=(4, 4, 4), stride=(1, 2, 2), norm=nn.BatchNorm3d, transposed=True),  # 1/2
            ConvBlock3D(conv_channels * 2,     conv_channels,     ksize=(4, 4, 4), stride=(2, 2, 2), norm=nn.BatchNorm3d, transposed=True)]) # 1

        self.final_rgba = ConvBlock3D(conv_channels, 4, ksize=(4, 4, 4), stride=(1, 1, 1), activation='tanh')
        self.final_flow = ConvBlock3D(conv_channels, 2, ksize=(4, 4, 4), stride=(1, 1, 1), activation='none')

        self.base_grid_bg_offset = None

    def render(self, x: torch.Tensor, global_context: GlobalContextVolume3D):
        """
        Pass inputs of a single layer through the network

        Parameters:
            x (torch.Tensor):       sampled texture concatenated with person IDs
            context (torch.Tensor): a context tensor read from the attention memory of the corresponding object layer

        Returns RGBa for the input layer and the final feature maps.
        """
        global_features, x, skips = self.memory_reader(x, global_context)

        x = torch.cat((global_features, x), dim=1)

        # decoding
        for layer in self.decoder:          
            x = torch.cat((x, skips.pop()), 1)
            x = layer(x)

        # finalizing render
        rgba = self.final_rgba(x)
        flow = self.final_flow(x)

        return rgba, flow


    def forward(self, input: dict) -> dict:
        """
        Apply forward pass of network

        Args:
            input (dict): collection of inputs to the network
        """
        query_input          = input["query_input"]
        spatiotemporal_noise = input["spatiotemporal_noise"][0]
        background_flow      = input["background_flow"]
        background_uv_map    = input["background_uv_map"]
        jitter_grid          = input["jitter_grid"]
        index                = input["index"]

        B, L, C, T, H, W = query_input.shape

        global_context = self.memory_encoder(spatiotemporal_noise.unsqueeze(0))

        composite_rgba = None
        composite_flow = background_flow

        layers_rgba = []
        layers_flow = []

        # camera stabilization correction
        index = index.transpose(0, 1).reshape(-1)
        background_offset = self.get_background_offset(jitter_grid, index)
        brightness_scale  = self.get_brightness_scale(jitter_grid, index) 

        for i in range(L):
            layer_input = query_input[:, i]

            # Background layer
            if i == 0:

                rgba, flow = self.render(layer_input, global_context)
                alpha = self.get_alpha_from_rgba(rgba)

                rgba = F.grid_sample(rgba, background_uv_map)               
                if self.do_adjustment:
                    rgba = self._apply_background_offset(rgba, background_offset)

                composite_rgba = rgba
                flow = composite_flow

            # Object layers
            else:
                rgba, flow = self.render(layer_input, global_context)
                alpha = self.get_alpha_from_rgba(rgba)

                composite_rgba = self.composite_rgba(rgba, rgba)
                composite_flow = flow * alpha + composite_flow * (1. - alpha)

            layers_rgba.append(rgba)
            layers_flow.append(flow)

        if self.do_adjustment:
            # map output to [0, 1]
            composite_rgba = composite_rgba * 0.5 + 0.5

            # adjust for brightness
            composite_rgba = torch.clamp(brightness_scale * composite_rgba, 0, 1)

            # map back to [-1, 1]
            composite_rgba = composite_rgba * 2 - 1

        # stack in layer dimension
        layers_rgba = torch.stack(layers_rgba, 1)
        layers_flow = torch.stack(layers_flow, 1)

        out = {
            "rgba_reconstruction": composite_rgba,          # [B,    4, T, H, W]
            "flow_reconstruction": composite_flow,          # [B,    2, T, H, w]
            "layers_rgba": layers_rgba,                     # [B, L, 4, T, H, W]
            "layers_flow": layers_flow,                     # [B, L, 2, T, H, W]
            "brightness_scale": brightness_scale,           # [B,    1, T, H, W]
            "background_offset": background_offset,         # [B,    2, T, H, W]
        }
        return out

    def get_background_offset(self, jitter_grid: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Get the background offset of the current set of frames

        Args:
            jitter_grid (torch.Tensor): sampling grid to apply jitter to the offset
            index (torch.Tensor):       tensor containing relevant frame indices
        """
        background_offset = F.interpolate(self.bg_offset, (self.max_frames, 4, 7), mode="trilinear")
        background_offset = background_offset[:, :, index].repeat(jitter_grid.shape[0], 1, 1, 1, 1)
        background_offset = F.grid_sample(background_offset, jitter_grid.permute(0, 2, 3, 4, 1), align_corners=True)
        
        # There is no offset in temporal dimension, so add zeros tensor
        background_offset = torch.cat((torch.zeros_like(background_offset[:, 0:1]), background_offset), dim=1)

        return background_offset

    def get_brightness_scale(self, jitter_grid: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Get the brightness scaling tensor of the current set of frames

        Args:
            jitter_grid (torch.Tensor): sampling grid to apply jitter to the offset
            index (torch.Tensor):       tensor containing relevant frame indices
        """

        brightness_scale = F.interpolate(self.brightness_scale, (self.max_frames, 4, 7), mode='trilinear', align_corners=True)
        brightness_scale = brightness_scale[:, :, index].repeat(jitter_grid.shape[0], 1, 1, 1, 1)
        brightness_scale = F.grid_sample(brightness_scale, jitter_grid.permute(0, 2, 3, 4, 1), align_corners=True)

        return brightness_scale

    def _apply_background_offset(self, input: torch.Tensor, bg_offset: torch.Tensor) -> torch.Tensor:
        """
        Apply the background offset to the input by constructing a base grid and adding the offset as a delta grid
        to get a sampling grid that is usable with F.grid_sample

        Args:
            input (torch.Tensor)
            bg_offset (torch.Tensor)

        Returns a resampling of the input tensor
        """

        batch_size, _, t, h, w = input.size()

        if self.base_grid_bg_offset == None:
            # Calculate a base grid that functions as an identity sampler
            temporal   = torch.linspace(-1.0, 1.0, t).view(1, 1, t, 1, 1).expand(batch_size, 1, t, h, w)
            horizontal = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, 1, w).expand(batch_size, 1, t, h, w)
            vertical   = torch.linspace(-1.0, 1.0, h).view(1, 1, 1, h, 1).expand(batch_size, 1, t, h, w)
            self.base_grid_bg_offset  = torch.cat([temporal, horizontal, vertical], dim=1).to(input.device)

        # current batch size may be smaller than normal batch size at the end of an epoch
        base_grid = self.base_grid_bg_offset[:batch_size]

        # calculate a Delta grid based on the flow field that offsets the base grid
        flow_grid = torch.cat([bg_offset[:, 0:1],
                               bg_offset[:, 1:2] / (w - 1.) / 2., 
                               bg_offset[:, 2:3] / (h - 1.) / 2.], 1)

        # construct full grid
        grid = (base_grid + flow_grid).permute(0, 2, 3, 4, 1)

        return F.grid_sample(input, grid, align_corners=True)


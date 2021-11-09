import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from InputProcessing.flowHandler import FlowHandler

from models.DynamicLayerDecomposition.modules.base_modules import ConvBlock
from models.DynamicLayerDecomposition.modules.base_modules import GlobalContextVolume
from models.DynamicLayerDecomposition.modules.base_modules import MemoryEncoder
from models.DynamicLayerDecomposition.modules.base_modules import MemoryReader
from models.DynamicLayerDecomposition.modules.base_modules import KeyValueEncoder
from models.DynamicLayerDecomposition.modules.base_modules import LayerDecompositionAttentionMemoryNet


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
    def __init__(self, keydim: int, valdim: int) -> None:
        super().__init__(keydim, valdim)

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

        return context_dist


class MemoryEncoder2D(MemoryEncoder):
    """
    Memory Encoder usable with 2D convolutions
    """
    def __init__(self, conv_channels: int, keydim: int, valdim: int, backbone: nn.Module) -> None:
        super().__init__(keydim, valdim)

        self.memory_encoder = KeyValueEncoder(nn.Conv2d, conv_channels, keydim, valdim, backbone)
        self.global_context = GlobalContextVolume2D(keydim, valdim)

    def _get_input(self, spatiotemporal_noise: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """
        Get the input for a single timestep to add to the global context

        Args:
            spatiotemoral_noise (torch.Tensor): noise tensor of the entire video
            frame_idx (int): index of first frame in the input
        """
        return spatiotemporal_noise[:, frame_idx].unsqueeze(0)

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


class LayerDecompositionAttentionMemoryNet2D(LayerDecompositionAttentionMemoryNet):
    """
    Layer Decomposition Attention Memory Net with 2D convolutions
    """
    def __init__(self, conv_channels=64, in_channels=16, valdim=128, keydim=64, max_frames=200, coarseness=10, do_adjustment=True, shared_encoder=True):
        super().__init__(max_frames, coarseness, do_adjustment)

        # initialize foreground encoder and decoder
        query_backbone = nn.ModuleList([
            ConvBlock2D(in_channels,       conv_channels,     ksize=4, stride=2),                                                  # 1/2
            ConvBlock2D(conv_channels,     conv_channels * 2, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/4
            ConvBlock2D(conv_channels * 2, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/8
            ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/16
            ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky'),  # 1/16
            ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky')]) # 1/16
                
        self.memory_reader = MemoryReader(nn.Conv2d, conv_channels * 4, keydim, valdim, query_backbone)

        if shared_encoder:
            self.memory_encoder = MemoryEncoder2D(conv_channels * 4, keydim, valdim, query_backbone)
        else:
            memory_backbone = nn.ModuleList([
                ConvBlock2D(in_channels,       conv_channels,     ksize=4, stride=2),                                                  # 1/2
                ConvBlock2D(conv_channels,     conv_channels * 2, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/4
                ConvBlock2D(conv_channels * 2, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/8
                ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/16
                ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky'),  # 1/16
                ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky')]) # 1/16

            self.memory_encoder = MemoryEncoder2D(conv_channels * 4, keydim, valdim, memory_backbone)

        decoder_in_channels = conv_channels * 4 + valdim * 2

        self.decoder = nn.ModuleList([
            ConvBlock2D(decoder_in_channels,   conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),  # 1/8
            ConvBlock2D(conv_channels * 2 * 4, conv_channels * 2, ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),  # 1/4
            ConvBlock2D(conv_channels * 2 * 2, conv_channels,     ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),  # 1/2
            ConvBlock2D(conv_channels * 2,     conv_channels,     ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True)]) # 1

        self.final_rgba = ConvBlock2D(conv_channels, 4, ksize=4, stride=1, activation='tanh')
        self.final_flow = ConvBlock2D(conv_channels, 2, ksize=4, stride=1, activation='none')
        
    def render(self, x: torch.Tensor, global_context: GlobalContextVolume, is_bg=False):
        """
        Pass inputs of a single layer through the network

        Parameters:
            x (torch.Tensor):       sampled texture concatenated with person IDs
            context (torch.Tensor): a context tensor read from the attention memory of the corresponding object layer

        Returns RGBa for the input layer and the final feature maps.
        """
        global_features, x, skips = self.memory_reader(x, global_context)

        if is_bg:
            x = torch.cat((torch.zeros_like(global_features), x), dim=1)
        else:
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
        # Get model input
        query_input          = input["query_input"]
        spatiotemporal_noise = input["spatiotemporal_noise"][0]
        background_flow      = input["background_flow"]
        background_uv_map    = input["background_uv_map"]
        jitter_grid          = input["jitter_grid"]
        index                = input["index"]

        B, L, C, T, H, W = query_input.shape

        query_input = self.reorder_time2batch(query_input)
        background_uv_map = background_uv_map.view(T*B, H, W, 2)

        composite_rgba = None
        composite_flow = self.reorder_time2batch(background_flow)

        layers_rgba = []
        layers_flow = []

        global_context = self.memory_encoder(spatiotemporal_noise)

        # For temporal consistency
        layers_alpha_warped = []
        composite_warped = None

        # camera stabilization correction
        index = index.transpose(0, 1).reshape(-1)
        jitter_grid = self.reorder_time2batch(jitter_grid)

        background_offset = self.get_background_offset(jitter_grid, index)
        brightness_scale  = self.get_brightness_scale(jitter_grid, index) 

        for i in range(L):
            layer_input = query_input[:, i]

            # Background layer
            if i == 0:
                rgba, flow = self.render(layer_input, global_context, is_bg=True)

                rgba = F.grid_sample(rgba, background_uv_map)               
                if self.do_adjustment:
                    rgba = FlowHandler.apply_flow(rgba, background_offset)

                composite_rgba = rgba
                flow = composite_flow

                # Temporal consistency 
                rgba_warped      = rgba[:B]
                composite_warped = rgba_warped[:, :3]
            # Object layers
            else:
                rgba, flow = self.render(layer_input, global_context)
                alpha = self.get_alpha_from_rgba(rgba)

                composite_rgba = self.composite_rgba(composite_rgba, rgba)
                composite_flow = flow * alpha + composite_flow * (1. - alpha)

                # Temporal consistency
                rgba_t1          = rgba[B:]
                rgba_warped      = FlowHandler.apply_flow(rgba_t1, flow[:B])
                alpha_warped     = self.get_alpha_from_rgba(rgba_warped)
                composite_warped = rgba_warped[:, :3] * alpha_warped + composite_warped * (1.0 - alpha_warped)
                composite_warped = self.composite_rgb(composite_warped, rgba_warped[:, :3], alpha_warped)

            layers_rgba.append(rgba)
            layers_flow.append(flow)
            layers_alpha_warped.append(rgba_warped[:, 3:4])

        if self.do_adjustment:
            # map output to [0, 1]
            composite_rgba = composite_rgba * 0.5 + 0.5

            # adjust for brightness
            composite_rgba = torch.clamp(brightness_scale * composite_rgba, 0, 1)

            # map back to [-1, 1]
            composite_rgba = composite_rgba * 2 - 1

        # stack in time dimension
        composite_rgba      = torch.stack((composite_rgba[:B], composite_rgba[B:]), -3)
        composite_flow      = torch.stack((composite_flow[:B], composite_flow[B:]), -3)
        layers_rgba         = torch.stack(layers_rgba, 1)
        layers_rgba         = torch.stack((layers_rgba[:B], layers_rgba[B:]), -3)
        layers_flow         = torch.stack(layers_flow, 1)
        layers_flow         = torch.stack((layers_flow[:B], layers_flow[B:]), -3)
        layers_alpha_warped = torch.stack(layers_alpha_warped, 1)
        brightness_scale    = torch.stack((brightness_scale[:B], brightness_scale[B:]), -3)
        background_offset   = torch.stack((background_offset[:B], background_offset[B:]), -3)

        out = {
            "rgba_reconstruction": composite_rgba,          # [B, 4, 2, H, W]
            "flow_reconstruction": composite_flow,          # [B, 2, 2, H, w]
            "reconstruction_warped": composite_warped,      # [B, 3, H, W]
            "layers_rgba": layers_rgba,                     # [B, L, 4, 2, H, W]
            "layers_flow": layers_flow,                     # [B, L, 2, 2, H, W]
            "layers_alpha_warped": layers_alpha_warped,     # [B, L, 1, H, W]
            "brightness_scale": brightness_scale,           # [B, 1, 2, H, W]
            "background_offset": background_offset,         # [B, 2, 2, H, W]
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
        background_offset = background_offset[0, :, index].transpose(0, 1)
        background_offset = F.grid_sample(background_offset, jitter_grid.permute(0, 2, 3, 1), align_corners=True)

        return background_offset

    def get_brightness_scale(self, jitter_grid: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Get the brightness scaling tensor of the current set of frames

        Args:
            jitter_grid (torch.Tensor): sampling grid to apply jitter to the offset
            index (torch.Tensor):       tensor containing relevant frame indices
        """
        brightness_scale = F.interpolate(self.brightness_scale, (self.max_frames, 4, 7), mode='trilinear', align_corners=True)
        brightness_scale = brightness_scale[0, 0, index].unsqueeze(1)
        brightness_scale = F.grid_sample(brightness_scale, jitter_grid.permute(0, 2, 3, 1), align_corners=True)

        return brightness_scale

    def reorder_time2batch(self, input: torch.Tensor) -> torch.Tensor:
        """
        Reorder the input tensor such that the time dimension is reversably stacked in the batch dimension
        
        Args:
            input (torch.Tensor)
        """
        if len(input.shape) == 6:
            b, l, c, t, h, w = input.shape
            return input.permute(0, 3, 1, 2, 4, 5).reshape(b*t, l, c, h, w)
        elif len(input.shape) == 5:
            b, c, t, h, w = input.shape
            return input.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
        else:
            raise NotImplementedError("Input shape is not supported")
import torch
import torch.nn as nn
import torch.nn.functional as F

from InputProcessing.flowHandler import FlowHandler
from models.DynamicLayerDecomposition.modules.base_modules import *
from models.DynamicLayerDecomposition.modules.modules_2d import *
from models.DynamicLayerDecomposition.modules.modules_3d import *


class Omnimatte(nn.Module):
    def __init__(self, conv_channels=64, in_channels=16, max_frames=200, coarseness=10, do_adjustment=True):
        super().__init__()
        self.encoder = nn.ModuleList([
            ConvBlock2D(in_channels,       conv_channels,     ksize=4, stride=2),
            ConvBlock2D(conv_channels,     conv_channels * 2, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock2D(conv_channels * 2, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky')])
        
        self.decoder = nn.ModuleList([
            ConvBlock2D(conv_channels * 4 * 2, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),
            ConvBlock2D(conv_channels * 4 * 2, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),
            ConvBlock2D(conv_channels * 4 * 2, conv_channels * 2, ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),
            ConvBlock2D(conv_channels * 2 * 2, conv_channels,     ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),
            ConvBlock2D(conv_channels * 2,     conv_channels,     ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True)])
        
        self.final_rgba = ConvBlock2D(conv_channels, 4, ksize=4, stride=1, activation='tanh')
        self.final_flow = ConvBlock2D(conv_channels, 2, ksize=4, stride=1, activation='tanh')

        self.bg_offset        = nn.Parameter(torch.zeros(1, 2, max_frames // coarseness, 4, 7))
        self.brightness_scale = nn.Parameter(torch.ones(1, 1, max_frames // coarseness, 4, 7))

        self.max_frames = max_frames
        self.do_adjustment = do_adjustment
        
    def render(self, x):
        """Pass inputs for a single layer through UNet.

        Parameters:
            x (tensor): sampled texture concatenated with person IDs

        Returns RGBa for the input layer and the final feature maps.
        """
        
        # encode
        skips = [x]
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < 5:
                skips.append(x)
        
        # decode
        for layer in self.decoder:
            x = torch.cat((x, skips.pop()), 1)
            x = layer(x)
        
        # finalize render
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
        input_tensor      = input["query_input"]
        background_flow   = input["background_flow"]
        background_uv_map = input["background_uv_map"]
        jitter_grid       = input["jitter_grid"]
        index             = input["index"]

        B, L, C, T, H, W = input_tensor.shape

        input_tensor = self.reorder_time2batch(input_tensor)
        background_uv_map = background_uv_map.view(T*B, H, W, 2)

        composite_rgba = None
        composite_flow = self.reorder_time2batch(background_flow)

        layers_rgba = []
        layers_flow = []

        # For temporal consistency
        layers_alpha_warped = []
        composite_warped = None

        # camera stabilization correction
        index = index.transpose(0, 1).reshape(-1)
        jitter_grid = self.reorder_time2batch(jitter_grid)

        background_offset = self.get_background_offset(jitter_grid, index)
        brightness_scale  = self.get_brightness_scale(jitter_grid, index) 

        for i in range(L):

            layer_input = input_tensor[:, i]

            rgba, flow = self.render(layer_input)
            alpha = self.get_alpha_from_rgba(rgba)

            # Static Background layer
            if i == 0:
                rgba = F.grid_sample(rgba, background_uv_map, align_corners=True)
                if self.do_adjustment:
                    rgba = FlowHandler.apply_flow(rgba, background_offset)

                composite_rgba = rgba
                flow = composite_flow

                # Temporal consistency 
                rgba_warped      = rgba[:B]
                composite_warped = rgba_warped[:, :3]
            # Omnimatte doesn't have a dynamic background layer
            if i == 1:
                rgba        = -1 * torch.ones_like(rgba)
                flow        = -1 * torch.ones_like(flow)
                rgba_warped = -1 * torch.ones_like(rgba[:B])
            # Object layers
            else:
                composite_rgba = rgba * alpha + composite_rgba * (1. - alpha)
                composite_flow = flow * alpha + composite_flow * (1. - alpha)

                # Temporal consistency
                rgba_t1          = rgba[B:]
                rgba_warped      = FlowHandler.apply_flow(rgba_t1, flow[:B])
                alpha_warped     = self.get_alpha_from_rgba(rgba_warped)
                composite_warped = rgba_warped[:, :3] * alpha_warped + composite_warped * (1.0 - alpha_warped)

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
            "rgba_reconstruction": composite_rgba,          # [B,    4, 2, H, W]
            "flow_reconstruction": composite_flow,          # [B,    2, 2, H, w]
            "reconstruction_warped": composite_warped,      # [B,       3, H, W]
            "layers_rgba": layers_rgba,                     # [B, L, 4, 2, H, W]
            "layers_flow": layers_flow,                     # [B, L, 2, 2, H, W]
            "layers_alpha_warped": layers_alpha_warped,     # [B, L,    1, H, W]
            "brightness_scale": brightness_scale,           # [B,    1, 2, H, W]
            "background_offset": background_offset,         # [B,    2, 2, H, W]
        }
        return out

    def get_alpha_from_rgba(self, rgba):
        """
        Get the alpha layer from an rgba tensor that is ready for compositing
        """
        return rgba[:, 3:4] * .5 + .5

    def get_background_offset(self, jitter_grid, index):
        background_offset = F.interpolate(self.bg_offset, (self.max_frames, 4, 7), mode="trilinear")
        background_offset = background_offset[0, :, index].transpose(0, 1)
        background_offset = F.grid_sample(background_offset, jitter_grid.permute(0, 2, 3, 1), align_corners=True)

        return background_offset

    def get_brightness_scale(self, jitter_grid, index):
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


class LayerDecompositionAttentionMemoryNet2D(LayerDecompositionAttentionMemoryNet):
    """
    Layer Decomposition Attention Memory Net with 2D convolutions
    """
    def __init__(self, flow_max, conv_channels=64, in_channels=16, valdim=128, keydim=64, max_frames=200, coarseness=10, do_adjustment=True, shared_encoder=True):
        super().__init__(max_frames, coarseness, do_adjustment)

        self.flow_max = flow_max

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
            self.memory_encoder = MemoryEncoder2D(conv_channels * 4, keydim, valdim, query_backbone, GlobalContextVolume2D)
        else:
            memory_backbone = nn.ModuleList([
                ConvBlock2D(in_channels,       conv_channels,     ksize=4, stride=2),                                                  # 1/2
                ConvBlock2D(conv_channels,     conv_channels * 2, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/4
                ConvBlock2D(conv_channels * 2, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/8
                ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/16
                ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky'),  # 1/16
                ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky')]) # 1/16

            self.memory_encoder = MemoryEncoder2D(conv_channels * 4, keydim, valdim, memory_backbone, GlobalContextVolume2D)

        decoder_in_channels = conv_channels * 4 + valdim * 2

        self.decoder = nn.ModuleList([
            ConvBlock2D(decoder_in_channels,   conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),  # 1/8
            ConvBlock2D(conv_channels * 2 * 4, conv_channels * 2, ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),  # 1/4
            ConvBlock2D(conv_channels * 2 * 2, conv_channels,     ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),  # 1/2
            ConvBlock2D(conv_channels * 2,     conv_channels,     ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True)]) # 1

        self.final_rgba = ConvBlock2D(conv_channels, 4, ksize=4, stride=1, activation='tanh')
        self.final_flow = ConvBlock2D(conv_channels, 2, ksize=4, stride=1, activation='tanh')
        
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
        query_input       = input["query_input"]
        memory_input      = input["memory_input"][0]
        background_flow   = input["background_flow"]
        background_uv_map = input["background_uv_map"]
        jitter_grid       = input["jitter_grid"]
        index             = input["index"]

        B, L, C, T, H, W = query_input.shape

        query_input = self.reorder_time2batch(query_input)
        background_uv_map = background_uv_map.view(T*B, H, W, 2)

        composite_rgba = None
        composite_flow = self.reorder_time2batch(background_flow)

        layers_rgba = []
        layers_flow = []

        global_context = self.memory_encoder(memory_input)

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

                rgba = F.grid_sample(rgba, background_uv_map, align_corners=True)
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
                rgba_warped      = FlowHandler.apply_flow(rgba_t1, flow[:B] * self.flow_max)
                alpha_warped     = self.get_alpha_from_rgba(rgba_warped)
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
            "rgba_reconstruction": composite_rgba,          # [B,    4, 2, H, W]
            "flow_reconstruction": composite_flow,          # [B,    2, 2, H, w]
            "reconstruction_warped": composite_warped,      # [B,       3, H, W]
            "layers_rgba": layers_rgba,                     # [B, L, 4, 2, H, W]
            "layers_flow": layers_flow,                     # [B, L, 2, 2, H, W]
            "layers_alpha_warped": layers_alpha_warped,     # [B, L,    1, H, W]
            "brightness_scale": brightness_scale,           # [B,    1, 2, H, W]
            "background_offset": background_offset,         # [B,    2, 2, H, W]
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
            self.memory_encoder = MemoryEncoder3D(conv_channels * 4, keydim, valdim, query_backbone, GlobalContextVolume3D, mem_freq, timesteps)
        else:
            memory_backbone = nn.ModuleList([
                ConvBlock3D(in_channels,       conv_channels,     ksize=(4, 4, 4), stride=(2, 2, 2)),                                                          # 1/2
                ConvBlock3D(conv_channels,     conv_channels * 2, ksize=(4, 4, 4), stride=(2, 2, 2),                norm=nn.BatchNorm3d, activation='leaky'),  # 1/4
                ConvBlock3D(conv_channels * 2, conv_channels * 4, ksize=(4, 4, 4), stride=(2, 2, 2),                norm=nn.BatchNorm3d, activation='leaky'),  # 1/8
                ConvBlock3D(conv_channels * 4, conv_channels * 4, ksize=(4, 4, 4), stride=(2, 2, 2),                norm=nn.BatchNorm3d, activation='leaky'),  # 1/16
                ConvBlock3D(conv_channels * 4, conv_channels * 4, ksize=(4, 4, 4), stride=(1, 1, 1), dil=(2, 2, 2), norm=nn.BatchNorm3d, activation='leaky'),  # 1/16
                ConvBlock3D(conv_channels * 4, conv_channels * 4, ksize=(4, 4, 4), stride=(1, 1, 1), dil=(2, 2, 2), norm=nn.BatchNorm3d, activation='leaky')]) # 1/16

            self.memory_encoder = MemoryEncoder3D(conv_channels * 4, keydim, valdim, memory_backbone, GlobalContextVolume3D, mem_freq, timesteps)

        decoder_in_channels = conv_channels * 4 + valdim * 2

        self.decoder = nn.ModuleList([
            ConvBlock3D(decoder_in_channels,   conv_channels * 4, ksize=(4, 4, 4), stride=(1, 2, 2), norm=nn.BatchNorm3d, transposed=True),  # 1/8
            ConvBlock3D(conv_channels * 2 * 4, conv_channels * 2, ksize=(4, 4, 4), stride=(2, 2, 2), norm=nn.BatchNorm3d, transposed=True),  # 1/4
            ConvBlock3D(conv_channels * 2 * 2, conv_channels,     ksize=(4, 4, 4), stride=(1, 2, 2), norm=nn.BatchNorm3d, transposed=True),  # 1/2
            ConvBlock3D(conv_channels * 2,     conv_channels,     ksize=(4, 4, 4), stride=(2, 2, 2), norm=nn.BatchNorm3d, transposed=True)]) # 1

        self.final_rgba = ConvBlock3D(conv_channels, 4, ksize=(4, 4, 4), stride=(1, 1, 1), activation='tanh')
        self.final_flow = ConvBlock3D(conv_channels, 2, ksize=(4, 4, 4), stride=(1, 1, 1), activation='tanh')

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
        query_input       = input["query_input"]
        memory_input      = input["memory_input"][0]
        background_flow   = input["background_flow"]
        background_uv_map = input["background_uv_map"]
        jitter_grid       = input["jitter_grid"]
        index             = input["index"]

        B, L, C, T, H, W = query_input.shape

        global_context = self.memory_encoder(memory_input.unsqueeze(0))

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
                rgba = F.grid_sample(rgba, background_uv_map, align_corners=True)               
                if self.do_adjustment:
                    rgba = self._apply_background_offset(rgba, background_offset)

                composite_rgba = rgba
                flow = composite_flow

            # Object layers
            else:
                rgba, flow = self.render(layer_input, global_context)
                alpha = self.get_alpha_from_rgba(rgba)

                composite_rgba = self.composite_rgba(composite_rgba, rgba)
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
            vertical   = torch.linspace(-1.0, 1.0, h).view(1, 1, 1, h, 1).expand(batch_size, 1, t, h, w)
            horizontal = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, 1, w).expand(batch_size, 1, t, h, w)
            self.base_grid_bg_offset  = torch.cat([horizontal, vertical, temporal], dim=1).to(input.device)

        # current batch size may be smaller than normal batch size at the end of an epoch
        base_grid = self.base_grid_bg_offset[:batch_size]

        # calculate a Delta grid based on the flow field that offsets the base grid
        flow_grid = torch.cat([bg_offset[:, 0:1],
                               bg_offset[:, 1:2] / (w - 1.) / 2., 
                               bg_offset[:, 2:3] / (h - 1.) / 2.], 1)

        # construct full grid
        grid = (base_grid + flow_grid).permute(0, 2, 3, 4, 1)

        return F.grid_sample(input, grid, align_corners=True)


class LayerDecompositionAttentionMemoryNetCombined(LayerDecompositionAttentionMemoryNet):
    """
    Layer Decomposition Attention Memory Net with 3D convolutions in the memory encoder and 2d Convolutions in the query encoder
    """
    def __init__(self, conv_channels=64, in_channels=16, valdim=128, keydim=64, max_frames=200, coarseness=10, mem_freq=4, timesteps=16, do_adjustment=True, gt_in_memory: bool=False):
        super().__init__(max_frames, coarseness, do_adjustment)

        if gt_in_memory:
            memory_in_channels = 5
        else:
            memory_in_channels = in_channels

        # initialize foreground encoder and decoder
        query_backbone = nn.ModuleList([
            ConvBlock2D(in_channels,       conv_channels,     ksize=4, stride=2),                                                  # 1/2
            ConvBlock2D(conv_channels,     conv_channels * 2, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/4
            ConvBlock2D(conv_channels * 2, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/8
            ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/16
            ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky'),  # 1/16
            ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky')]) # 1/16
                
        self.memory_reader = MemoryReader(nn.Conv2d, conv_channels * 4, keydim, valdim, query_backbone)

        memory_backbone = nn.ModuleList([
        ConvBlock3D(memory_in_channels, conv_channels,     ksize=(4, 4, 4), stride=(2, 2, 2)),                                                          # 1/2
        ConvBlock3D(conv_channels,      conv_channels * 2, ksize=(4, 4, 4), stride=(2, 2, 2),                norm=nn.BatchNorm3d, activation='leaky'),  # 1/4
        ConvBlock3D(conv_channels * 2,  conv_channels * 4, ksize=(4, 4, 4), stride=(2, 2, 2),                norm=nn.BatchNorm3d, activation='leaky'),  # 1/8
        ConvBlock3D(conv_channels * 4,  conv_channels * 4, ksize=(4, 4, 4), stride=(2, 2, 2),                norm=nn.BatchNorm3d, activation='leaky'),  # 1/16
        ConvBlock3D(conv_channels * 4,  conv_channels * 4, ksize=(4, 4, 4), stride=(1, 1, 1), dil=(1, 2, 2), norm=nn.BatchNorm3d, activation='leaky'),  # 1/16
        ConvBlock3D(conv_channels * 4,  conv_channels * 4, ksize=(4, 4, 4), stride=(1, 1, 1), dil=(1, 2, 2), norm=nn.BatchNorm3d, activation='leaky')]) # 1/16

        self.memory_encoder = MemoryEncoder3D(conv_channels * 4, keydim, valdim, memory_backbone, GlobalContextVolume2D, mem_freq=mem_freq, timesteps=timesteps)

        self.dynamics_layer = ConvBlock3D(valdim * 2, valdim, ksize=(4, 4, 4), stride=(1, 1, 1), norm=nn.BatchNorm3d, transposed=True)

        self.decoder = nn.ModuleList([
            ConvBlock2D(conv_channels * 4 + valdim, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),  # 1/8
            ConvBlock2D(conv_channels * 2 * 4, conv_channels * 2, ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),  # 1/4
            ConvBlock2D(conv_channels * 2 * 2, conv_channels,     ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),  # 1/2
            ConvBlock2D(conv_channels * 2,     conv_channels,     ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True)]) # 1

        self.final_rgba = ConvBlock2D(conv_channels, 4, ksize=4, stride=1, activation='tanh')
        self.final_flow = ConvBlock2D(conv_channels, 2, ksize=4, stride=1, activation='tanh')

        self.base_grid_bg_offset = None

    def render(self, x: torch.Tensor, global_context: GlobalContextVolume2D):
        """
        Pass inputs of a single layer through the network

        Parameters:
            x (torch.Tensor):       sampled texture concatenated with person IDs
            context (torch.Tensor): a context tensor read from the attention memory of the corresponding object layer

        Returns RGBa for the input layer and the final feature maps.
        """
        T = x.shape[-3]

        outputs = []
        skips = []
        for t in range(T):
            global_features, x_t, skips_t = self.memory_reader(x[..., t, :, :], global_context)

            x_t = torch.cat((global_features, x_t), dim=1)
            
            outputs.append(x_t)
            skips.append(skips_t)

        x = torch.stack(outputs, dim=-3)

        x = self.dynamics_layer(x)

        rgba = []
        flow = []
        for t in range(T):
            # decoding
            x_t = x[..., t, :, :]
            skips_t = skips[t]

            for layer in self.decoder:
                x_t = torch.cat((x_t, skips_t.pop()), 1)
                x_t = layer(x_t)
        
            # finalizing render
            rgba.append(self.final_rgba(x_t))
            flow.append(self.final_flow(x_t))

        rgba = torch.stack(rgba, dim=-3)
        flow = torch.stack(flow, dim=-3)

        return rgba, flow

    def render_background(self, x: torch.Tensor, global_context: GlobalContextVolume2D):
        """
        Pass inputs of a single layer through the network

        Parameters:
            x (torch.Tensor):       sampled texture concatenated with person IDs
            context (torch.Tensor): a context tensor read from the attention memory of the corresponding object layer

        Returns RGBa for the input layer and the final feature maps.
        """

        _, _, T, _, _ = x.shape

        rgba = []
        flow = []

        _, x, skips = self.memory_reader(x[:, :, 0], global_context)
        
        # decoding
        for layer in self.decoder:          
            x = torch.cat((x, skips.pop()), 1)
            x = layer(x)

        # finalizing render
        rgba = self.final_rgba(x).unsqueeze(2).repeat(1, 1, T, 1, 1)
        flow = self.final_flow(x).unsqueeze(2).repeat(1, 1, T, 1, 1)

        return rgba, flow


    def forward(self, input: dict) -> dict:
        """
        Apply forward pass of network

        Args:
            input (dict): collection of inputs to the network
        """
        query_input       = input["query_input"]
        memory_input      = input["memory_input"]
        background_flow   = input["background_flow"]
        background_uv_map = input["background_uv_map"]
        jitter_grid       = input["jitter_grid"]
        index             = input["index"]

        B, L, C, T, H, W = query_input.shape

        global_context = self.memory_encoder(memory_input)

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

                rgba, flow = self.render_background(layer_input, global_context)
                rgba = F.grid_sample(rgba, background_uv_map, align_corners=True)
                if self.do_adjustment:
                    rgba = self._apply_background_offset(rgba, background_offset)

                composite_rgba = rgba
                flow = composite_flow

            # Object layers
            else:
                rgba, flow = self.render(layer_input, global_context)
                alpha = self.get_alpha_from_rgba(rgba)

                composite_rgba = self.composite_rgba(composite_rgba, rgba)
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
            vertical   = torch.linspace(-1.0, 1.0, h).view(1, 1, 1, h, 1).expand(batch_size, 1, t, h, w)
            horizontal = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, 1, w).expand(batch_size, 1, t, h, w)
            self.base_grid_bg_offset  = torch.cat([horizontal, vertical, temporal], dim=1).to(input.device)

        # current batch size may be smaller than normal batch size at the end of an epoch
        base_grid = self.base_grid_bg_offset[:batch_size]

        # calculate a Delta grid based on the flow field that offsets the base grid
        flow_grid = torch.cat([bg_offset[:, 0:1],
                               bg_offset[:, 1:2] / (w - 1.) / 2., 
                               bg_offset[:, 2:3] / (h - 1.) / 2.], 1)

        # construct full grid
        grid = (base_grid + flow_grid).permute(0, 2, 3, 4, 1)

        return F.grid_sample(input, grid, align_corners=True)


class LayerDecompositionAttentionMemoryNet3DBottleneck(LayerDecompositionAttentionMemoryNet):
    """
    Layer Decomposition Attention Memory Net with 2D convolutions and one 3D convolution in the middle to function as a temporal bottleneck
    """
    def __init__(self, conv_channels=64, in_channels=16, valdim=128, keydim=64, max_frames=200, coarseness=10, do_adjustment=True, shared_encoder=True, gt_in_memory: bool=False):
        super().__init__(max_frames, coarseness, do_adjustment)

        if shared_encoder and gt_in_memory:
            raise ValueError("Cannot use both a shared memory encoder and ground truth in memory input.")
        elif gt_in_memory:
            memory_in_channels = 5
        else:
            memory_in_channels = in_channels

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
            self.memory_encoder = MemoryEncoder2D(conv_channels * 4, keydim, valdim, query_backbone, GlobalContextVolume2D)
        else:
            memory_backbone = nn.ModuleList([
            ConvBlock2D(memory_in_channels, conv_channels,     ksize=4, stride=2),                                                  # 1/2
            ConvBlock2D(conv_channels,      conv_channels * 2, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/4
            ConvBlock2D(conv_channels * 2,  conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/8
            ConvBlock2D(conv_channels * 4,  conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/16
            ConvBlock2D(conv_channels * 4,  conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky'),  # 1/16
            ConvBlock2D(conv_channels * 4,  conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky')]) # 1/16

            self.memory_encoder = MemoryEncoder2D(conv_channels * 4, keydim, valdim, memory_backbone, GlobalContextVolume2D)

        self.dynamics_layer = ConvBlock3D(valdim * 2, valdim, ksize=(4, 4, 4), stride=(1, 1, 1), norm=nn.BatchNorm3d, transposed=True)

        self.decoder = nn.ModuleList([
            ConvBlock2D(conv_channels * 4 + valdim, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),  # 1/8
            ConvBlock2D(conv_channels * 2 * 4, conv_channels * 2, ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),  # 1/4
            ConvBlock2D(conv_channels * 2 * 2, conv_channels,     ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True),  # 1/2
            ConvBlock2D(conv_channels * 2,     conv_channels,     ksize=4, stride=2, norm=nn.BatchNorm2d, transposed=True)]) # 1

        self.final_rgba = ConvBlock2D(conv_channels, 4, ksize=4, stride=1, activation='tanh')
        self.final_flow = ConvBlock2D(conv_channels, 2, ksize=4, stride=1, activation='tanh')

        self.base_grid_bg_offset = None

    def render(self, x: torch.Tensor, global_context: GlobalContextVolume2D):
        """
        Pass inputs of a single layer through the network

        Parameters:
            x (torch.Tensor):       sampled texture concatenated with person IDs
            context (torch.Tensor): a context tensor read from the attention memory of the corresponding object layer

        Returns RGBa for the input layer and the final feature maps.
        """
        T = x.shape[-3]

        outputs = []
        skips = []
        for t in range(T):
            global_features, x_t, skips_t = self.memory_reader(x[..., t, :, :], global_context)

            x_t = torch.cat((global_features, x_t), dim=1)
            
            outputs.append(x_t)
            skips.append(skips_t)

        x = torch.stack(outputs, dim=-3)

        x = self.dynamics_layer(x)

        rgba = []
        flow = []
        for t in range(T):
            # decoding
            x_t = x[..., t, :, :]
            skips_t = skips[t]

            for layer in self.decoder:
                x_t = torch.cat((x_t, skips_t.pop()), 1)
                x_t = layer(x_t)
        
            # finalizing render
            rgba.append(self.final_rgba(x_t))
            flow.append(self.final_flow(x_t))

        rgba = torch.stack(rgba, dim=-3)
        flow = torch.stack(flow, dim=-3)

        return rgba, flow

    def render_background(self, x: torch.Tensor, global_context: GlobalContextVolume2D):
        """
        Pass inputs of a single layer through the network

        Parameters:
            x (torch.Tensor):       sampled texture concatenated with person IDs
            context (torch.Tensor): a context tensor read from the attention memory of the corresponding object layer

        Returns RGBa for the input layer and the final feature maps.
        """

        _, _, T, _, _ = x.shape

        rgba = []
        flow = []

        _, x, skips = self.memory_reader(x[:, :, 0], global_context)
        
        # decoding
        for layer in self.decoder:          
            x = torch.cat((x, skips.pop()), 1)
            x = layer(x)

        # finalizing render
        rgba = self.final_rgba(x).unsqueeze(2).repeat(1, 1, T, 1, 1)
        flow = self.final_flow(x).unsqueeze(2).repeat(1, 1, T, 1, 1)

        return rgba, flow


    def forward(self, input: dict) -> dict:
        """
        Apply forward pass of network

        Args:
            input (dict): collection of inputs to the network
        """
        query_input       = input["query_input"]
        memory_input      = input["memory_input"][0]
        background_flow   = input["background_flow"]
        background_uv_map = input["background_uv_map"]
        jitter_grid       = input["jitter_grid"]
        index             = input["index"]

        B, L, C, T, H, W = query_input.shape

        global_context = self.memory_encoder(memory_input)

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

                rgba, flow = self.render_background(layer_input, global_context)
                rgba = F.grid_sample(rgba, background_uv_map, align_corners=True)
                if self.do_adjustment:
                    rgba = self._apply_background_offset(rgba, background_offset)

                composite_rgba = rgba
                flow = composite_flow

            # Object layers
            else:
                rgba, flow = self.render(layer_input, global_context)
                alpha = self.get_alpha_from_rgba(rgba)

                composite_rgba = self.composite_rgba(composite_rgba, rgba)
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
            vertical   = torch.linspace(-1.0, 1.0, h).view(1, 1, 1, h, 1).expand(batch_size, 1, t, h, w)
            horizontal = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, 1, w).expand(batch_size, 1, t, h, w)
            self.base_grid_bg_offset  = torch.cat([horizontal, vertical, temporal], dim=1).to(input.device)

        # current batch size may be smaller than normal batch size at the end of an epoch
        base_grid = self.base_grid_bg_offset[:batch_size]

        # calculate a Delta grid based on the flow field that offsets the base grid
        flow_grid = torch.cat([bg_offset[:, 0:1],
                               bg_offset[:, 1:2] / (w - 1.) / 2., 
                               bg_offset[:, 2:3] / (h - 1.) / 2.], 1)

        # construct full grid
        grid = (base_grid + flow_grid).permute(0, 2, 3, 4, 1)

        return F.grid_sample(input, grid, align_corners=True)

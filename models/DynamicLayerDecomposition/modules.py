import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from InputProcessing.flowHandler import FlowHandler

class ConvBlock(nn.Module):
    """Helper module consisting of a convolution, optional normalization and activation, with padding='same'.
    
    Adapted from Layered Neural Rendering (https://github.com/google/retiming)
    """

    def __init__(self, conv, in_channels, out_channels, ksize=4, stride=1, dil=1, norm=None, activation='relu'):
        """Create a conv block.

        Parameters:
            conv (convolutional layer) - - the type of conv layer, e.g. Conv2d, ConvTranspose2d
            in_channels (int) - - the number of input channels
            in_channels (int) - - the number of output channels
            ksize (int) - - the kernel size
            stride (int) - - stride
            dil (int) - - dilation
            norm (norm layer) - - the type of normalization layer, e.g. BatchNorm2d, InstanceNorm2d
            activation (str)  -- the type of activation: relu | leaky | tanh | none
        """
        super(ConvBlock, self).__init__()
        self.k = ksize
        self.s = stride
        self.d = dil
        self.conv = conv(in_channels, out_channels, ksize, stride=stride, dilation=dil)

        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = None

        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None

    def forward(self, x):
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

class KeyValueEncoder(nn.Module):

    def __init__(self, conv_channels: int, keydim: int, valdim: int, encoder: torch.nn.Module) -> None:
        super().__init__()

        self.encoder = encoder

        self.key_layer = nn.Conv2d(conv_channels, keydim, kernel_size=3, padding=1)
        self.val_layer = nn.Conv2d(conv_channels, valdim, kernel_size=3, padding=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        skips = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < 4:
                skips.append(x)

        key = self.key_layer(x)
        val = self.val_layer(x)
        
        return key, val, skips


class GlobalContextVolume(nn.Module):
    def __init__(self, keydim: int, valdim: int) -> None:
        super().__init__()

        self.register_buffer("context_volume", torch.zeros((valdim, keydim)))

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

    def update(self, local_contexts: list) -> None:
        """
        Update the global context volume using the local context matrices at different timesteps
        v1:
            the average of the local context matrices is taken
        
        Args:
            local_contexts (list[ torch.Tensor[C_v x C_k] ] -- length=T)
        """
        self.context_volume = torch.mean(torch.stack(local_contexts, dim=0), dim=0, keepdim=False)


class MemoryEncoder(nn.Module):

    def __init__(self, conv_channels: int, keydim: int, valdim: int, backbone: nn.Module) -> None:
        super().__init__()

        self.memory_encoder = KeyValueEncoder(conv_channels, keydim, valdim, backbone)
        self.global_context = GlobalContextVolume(keydim, valdim)

        self.valdim          = valdim
        self.keydim          = keydim

    def forward(self, spatiotemporal_noise: torch.Tensor) -> GlobalContextVolume:
        """
        Set the global context volumes from a set. This is equivalent to memorizing frames in a memory network
        but because we are overfitting on a single video we do this with a number of frames at the start of each epoch
        """
    
        local_contexts = []

        # add frames to memory
        for frame_idx in range(spatiotemporal_noise.shape[0]):

            input = spatiotemporal_noise[frame_idx:frame_idx+1]

            # encode frame and get context matrix
            key, value, _ = self.memory_encoder(input)
            context = self.get_context_from_key_value_pair(key, value)

            # remove batch dimension
            context = context.squeeze(0)

            # append to collection of local contexts of current layer
            local_contexts.append(context)

        # update the memory of the current layer
        self.global_context.update(local_contexts)

        return self.global_context

    def get_context_from_key_value_pair(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Takes as argument a key value pair and returns the corresponding context matrix

        The context is calculated according to:
            C [B x C_v x C_k] = v [B x C_v x HW] @ k [B x HW x C_k] 

        Args:
            key   (Tensor[B x C_k x H x W])
            value (Tensor[B x C_v x H x W])

        Returns:
            context (Tensor[B x C_v x C_k])
        """
        B, _, H, W = key.shape
        
        key   = key.view(B, -1, H*W).permute(0, 2, 1)
        value = value.view(B, -1, H*W)

        context = torch.matmul(value, key)

        return context


class MemoryReader(nn.Module):
    def __init__(self, conv_channels: int, keydim: int, valdim: int, backbone: nn.Module) -> None:
        super().__init__()

        self.query_encoder = KeyValueEncoder(conv_channels, keydim, valdim, backbone)

        self.keydim = keydim
        self.valdim = valdim

    def forward(self, input: torch.Tensor, global_context: GlobalContextVolume) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor[B x C x H x W])

        Returns:
            feature_map (torch.Tensor[B x 2 * C_v x H x W])
        """

        query, value, skips = self.query_encoder(input)

        global_features = global_context(query)

        return global_features, value, skips

class LayerDecompositionAttentionMemoryNet(nn.Module):
    def __init__(self, conv_channels=64, in_channels=16, valdim=128, keydim=64, max_frames=200, coarseness=10, do_adjustment=True, shared_encoder=True):
        super().__init__()

        # initialize foreground encoder and decoder
        self.query_backbone = nn.ModuleList([
            ConvBlock(nn.Conv2d, in_channels,       conv_channels,     ksize=4, stride=2),                                                  # 1/2
            ConvBlock(nn.Conv2d, conv_channels,     conv_channels * 2, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/4
            ConvBlock(nn.Conv2d, conv_channels * 2, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/8
            ConvBlock(nn.Conv2d, conv_channels * 4, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/16
            ConvBlock(nn.Conv2d, conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky'),  # 1/16
            ConvBlock(nn.Conv2d, conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky')]) # 1/16
                
        self.memory_reader = MemoryReader(conv_channels * 4, keydim, valdim, self.query_backbone)

        if shared_encoder:
            self.memory_encoder = MemoryEncoder(conv_channels * 4, keydim, valdim, self.query_backbone)
        else:
            self.memory_backbone = nn.ModuleList([
                ConvBlock(nn.Conv2d, in_channels,       conv_channels,     ksize=4, stride=2),                                                  # 1/2
                ConvBlock(nn.Conv2d, conv_channels,     conv_channels * 2, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/4
                ConvBlock(nn.Conv2d, conv_channels * 2, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/8
                ConvBlock(nn.Conv2d, conv_channels * 4, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),  # 1/16
                ConvBlock(nn.Conv2d, conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky'),  # 1/16
                ConvBlock(nn.Conv2d, conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky')]) # 1/16

            self.memory_encoder = MemoryEncoder(conv_channels * 4, keydim, valdim, self.memory_backbone)

        decoder_in_channels = conv_channels * 4 + valdim * 2

        self.decoder = nn.ModuleList([
            ConvBlock(nn.ConvTranspose2d, decoder_in_channels,   conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d),  # 1/8
            ConvBlock(nn.ConvTranspose2d, conv_channels * 2 * 4, conv_channels * 2, ksize=4, stride=2, norm=nn.BatchNorm2d),  # 1/4
            ConvBlock(nn.ConvTranspose2d, conv_channels * 2 * 2, conv_channels,     ksize=4, stride=2, norm=nn.BatchNorm2d),  # 1/2
            ConvBlock(nn.ConvTranspose2d, conv_channels * 2,     conv_channels,     ksize=4, stride=2, norm=nn.BatchNorm2d)]) # 1

        self.final_rgba = ConvBlock(nn.Conv2d, conv_channels, 4, ksize=4, stride=1, activation='tanh')
        self.final_flow = ConvBlock(nn.Conv2d, conv_channels, 2, ksize=4, stride=1, activation='none')

        self.bg_offset        = nn.Parameter(torch.zeros(1, 2, max_frames // coarseness, 4, 7))
        self.brightness_scale = nn.Parameter(torch.ones(1, 1, max_frames // coarseness, 4, 7))

        self.max_frames = max_frames
        self.do_adjustment = do_adjustment
        
    def render(self, x: torch.Tensor, global_context: GlobalContextVolume, is_bg=False):
        """Pass inputs for a single layer through UNet.

        Parameters:
            x (tensor) - - sampled texture concatenated with person IDs
            context (tensor) - - a context tensor read from the attention memory of the corresponding object layer

        Returns RGBA for the input layer and the final feature maps.
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


    def forward(self, input):
        """
        Apply forward pass of network
        """

        input_tensor         = input["input_tensor"]
        spatiotemporal_noise = input["spatiotemporal_noise"][0]
        background_flow      = input["background_flow"]
        background_uv_map    = input["background_uv_map"]
        jitter_grid          = input["jitter_grid"]
        index                = input["index"]

        global_context = self.memory_encoder(spatiotemporal_noise)

        batch_size, N_t, N_layers, channels, H, W = input_tensor.shape

        input_t0 = input_tensor[:, 0]
        input_t1 = input_tensor[:, 1]

        composite_rgba = None
        composite_flow = torch.cat((background_flow[:, 0], background_flow[:, 1]))

        background_uv_map = torch.cat((background_uv_map[:, 0], background_uv_map[:, 1]))

        layers_rgba = []
        layers_flow = []

        # For temporal consistency
        layers_alpha_warped = []
        composite_warped = None

        # camera stabilization correction
        index = index.transpose(0, 1).reshape(-1)
        jitter_grid = torch.cat((jitter_grid[:, 0], jitter_grid[:, 1]))
        background_offset = self.get_background_offset(jitter_grid, index)
        brightness_scale  = self.get_brightness_scale(jitter_grid, index) 

        for i in range(N_layers):
            layer_input = torch.cat(([input_t0[:, i], input_t1[:, i]]))

            # Background layer
            if i == 0:
                rgba, flow = self.render(layer_input, global_context, is_bg=True)
                alpha = self.get_alpha_from_rgba(rgba)

                rgba = F.grid_sample(rgba, background_uv_map)               
                if self.do_adjustment:
                    rgba = FlowHandler.apply_flow(rgba, background_offset)

                composite_rgba = rgba
                flow = composite_flow

                # Temporal consistency 
                rgba_warped      = rgba[:batch_size]
                composite_warped = rgba_warped[:, :3]
            # Object layers
            else:
                rgba, flow = self.render(layer_input, global_context)
                alpha = self.get_alpha_from_rgba(rgba)

                composite_rgba = rgba * alpha + composite_rgba * (1. - alpha)
                composite_flow = flow * alpha + composite_flow * (1. - alpha)

                # Temporal consistency
                rgba_t1          = rgba[batch_size:]
                rgba_warped      = FlowHandler.apply_flow(rgba_t1, flow[:batch_size])
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
        composite_rgba      = torch.stack((composite_rgba[:batch_size], composite_rgba[batch_size:]), 1)
        composite_flow      = torch.stack((composite_flow[:batch_size], composite_flow[batch_size:]), 1)
        layers_rgba         = torch.stack(layers_rgba, 1)
        layers_rgba         = torch.stack((layers_rgba[:batch_size], layers_rgba[batch_size:]), 1)
        layers_flow         = torch.stack(layers_flow, 1)
        layers_flow         = torch.stack((layers_flow[:batch_size], layers_flow[batch_size:]), 1)
        layers_alpha_warped = torch.stack(layers_alpha_warped, 1)
        brightness_scale    = torch.stack((brightness_scale[:batch_size], brightness_scale[batch_size:]), 1)
        background_offset   = torch.stack((background_offset[:batch_size], background_offset[batch_size:]), 1)

        out = {
            "rgba_reconstruction": composite_rgba,          # [B, 2, 4, H, W]
            "flow_reconstruction": composite_flow,          # [B, 2, 2, H, w]
            "reconstruction_warped": composite_warped,      # [B, 3, H, W]
            "layers_rgba": layers_rgba,                     # [B, 2, L, 4, H, W]
            "layers_flow": layers_flow,                     # [B, 2, L, 2, H, W]
            "layers_alpha_warped": layers_alpha_warped,     # [B, L, 1, H, W]
            "brightness_scale": brightness_scale,           # [B, 2, 1, H, W]
            "background_offset": background_offset,         # [B, 2, 2, H, W]
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